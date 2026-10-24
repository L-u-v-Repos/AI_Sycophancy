from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from typing import Any, Dict, List, Optional
import pandas as pd
from tqdm import tqdm

from .client import call_llm
from .config import (
    BELIEF_H_MAX,
    BELIEF_H_MIN,
    Scenario,
    SYSTEM_PROMPTS,
    SCENARIOS,
    DEFAULT_P_USER_GRID,
    CONDITIONS,
)
from .evaluator import evaluate_response
from .metrics import belief_probabilities_from_H, short_hash, stance_from_logodds
from .user_agent import ControlledUser, LLMUser, generate_user_message


def make_evidence_schedule(
    turns: int,
    schedule_type: str = "none",
    strength: float = 0.5,
) -> List[float]:
    # return external ground truth evidence strength for each turn
    if schedule_type == "none":
        return [0.0] * turns
    if schedule_type == "constant":
        return [strength] * turns
    if schedule_type == "step":
        # no evidence first half, then evidence second half
        half = turns // 2
        return [0.0] * half + [strength] * (turns - half)
    if schedule_type == "ramp":
        # linearly increase evidence from 0 to strength
        return [strength * (i / max(1, turns - 1)) for i in range(turns)]
    raise ValueError(f"Unknown evidence schedule type: '{schedule_type}'. Choose: none, constant, step, ramp")


def run_static_trial(
    client,
    assistant_model: str,
    evaluator_model: str,
    scenario: Scenario,
    condition: str,
    initial_belief_H: float,
    assistant_temperature: float = 0.0,
    evaluator_client=None,
    custom_user_message: Optional[str] = None,
    user_sycophancy_liking: str = "medium",
) -> Dict[str, Any]:
    if condition not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown condition '{condition}'. Must be one of: {list(SYSTEM_PROMPTS.keys())}")

    if not BELIEF_H_MIN <= initial_belief_H <= BELIEF_H_MAX:
        raise ValueError(f"initial_belief_H must be between {BELIEF_H_MIN:g} and {BELIEF_H_MAX:g}.")
    user_stance_val = initial_belief_H / BELIEF_H_MAX

    system_prompt = SYSTEM_PROMPTS[condition]
    if custom_user_message and custom_user_message.strip():
        user_msg = custom_user_message.strip()
    else:
        user_msg = generate_user_message(scenario, p_user=0.0, user_stance=user_stance_val)

    initial_eval = evaluate_response(
        client=evaluator_client if evaluator_client is not None else client,
        model=evaluator_model,
        dialogue_text=f"User: {user_msg}",
        ground_truth=scenario.ground_truth,
        false_belief=scenario.false_belief,
        latest_speaker="user",
    )

    # record version hashes for reproducibility
    scenario_hash = short_hash(scenario)
    sysprompt_hash = short_hash(system_prompt)
    userprompt_hash = short_hash(user_msg)

    # get assistant response with token usage
    assistant_response = call_llm(
        client=client,
        model=assistant_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=assistant_temperature,
    )
    if not assistant_response:
        raise RuntimeError(
            f"Assistant returned an empty response for scenario={scenario.id}, condition={condition}"
        )

    # score the assistant response with the evaluator judge
    eval_result = evaluate_response(
        client=evaluator_client if evaluator_client is not None else client,
        model=evaluator_model,
        dialogue_text=f"User: {user_msg}\nChatbot: {assistant_response}",
        ground_truth=scenario.ground_truth,
        false_belief=scenario.false_belief,
        latest_speaker="assistant",
    )

    return {
        "scenario_id": scenario.id,
        "domain": scenario.domain,
        "condition": condition,
        "initial_belief_setting": initial_belief_H,
        "initial_belief_H": initial_eval["H"],
        "user_H": initial_eval["H"],
        "user_p1": initial_eval["p1"],
        "user_p2": initial_eval["p2"],
        "user_p_false": initial_eval["p_false"],
        "user_p_truth": initial_eval["p_truth"],
        "user_H_rationale": initial_eval.get("rationale", ""),
        "initial_belief_p_false": initial_eval["p_false"],
        "initial_belief_p_truth": initial_eval["p_truth"],
        "initial_belief_p1": initial_eval["p1"],
        "initial_belief_p2": initial_eval["p2"],
        "initial_belief_rationale": initial_eval.get("rationale", ""),
        "H": eval_result["H"],
        "assistant_H": eval_result["H"],
        "assistant_p1": eval_result["p1"],
        "assistant_p2": eval_result["p2"],
        "assistant_p_false": eval_result["p_false"],
        "assistant_p_truth": eval_result["p_truth"],
        "assistant_H_rationale": eval_result.get("rationale", ""),
        "p_false": eval_result["p_false"],
        "p_truth": eval_result["p_truth"],
        "H_rationale": eval_result.get("rationale", ""),
        "user_msg": user_msg,
        "assistant_response": assistant_response,
        # versioning hashes
        "scenario_hash": scenario_hash,
        "sysprompt_hash": sysprompt_hash,
        "userprompt_hash": userprompt_hash,
        **eval_result,
    }


def build_static_jobs(
    scenarios: List[Scenario],
    conditions: List[str],
    p_user_grid: List[float],
) -> List[Dict[str, Any]]:
    # build the full cartesian product of scenario × condition × belief level
    jobs = []
    for sc in scenarios:
        for cond in conditions:
            for p in p_user_grid:
                jobs.append({"scenario": sc, "condition": cond, "p_user": p})
    return jobs


def run_static_experiment(
    client,
    assistant_model: str,
    evaluator_model: str,
    scenarios: List[Scenario],
    conditions: Optional[List[str]] = None,
    p_user_grid: Optional[List[float]] = None,
    max_workers: int = 2,
    checkpoint_path: Optional[str] = None,
    delay_between_calls: float = 0.5,
    evaluator_client=None,
) -> pd.DataFrame:
    if not scenarios:
        raise ValueError("No scenarios provided to run_static_experiment.")

    conditions = conditions if conditions is not None else CONDITIONS
    p_user_grid = p_user_grid if p_user_grid is not None else DEFAULT_P_USER_GRID

    jobs = build_static_jobs(scenarios, conditions, p_user_grid)
    records: List[Dict[str, Any]] = []
    completed_keys: set = set()

    # resume from checkpoint if it exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            existing_df = pd.read_csv(checkpoint_path)
            if "initial_belief_H" in existing_df.columns and "H" in existing_df.columns:
                records = existing_df.to_dict(orient="records")
                for r in records:
                    setting = r.get("initial_belief_setting", r["initial_belief_H"])
                    key = (r["scenario_id"], r["condition"], round(float(setting), 4))
                    completed_keys.add(key)
                print(f"  resuming from checkpoint: {len(records)} trials already done")
            else:
                print("  checkpoint uses the previous evaluator format; starting fresh")
        except Exception as e:
            print(f"  could not load checkpoint ({e}), starting fresh")

    pending_jobs = [
        j for j in jobs
        if (j["scenario"].id, j["condition"], round(float(j["p_user"]), 4)) not in completed_keys
    ]
    print(f"  {len(pending_jobs)} trials to run ({len(completed_keys)} already complete)")

    if not pending_jobs:
        return pd.DataFrame(records)

    def _execute(job: Dict[str, Any]) -> Dict[str, Any]:
        # small delay to stay within api rate limits
        time.sleep(delay_between_calls)
        return run_static_trial(
            client=client,
            assistant_model=assistant_model,
            evaluator_model=evaluator_model,
            scenario=job["scenario"],
            condition=job["condition"],
            initial_belief_H=job["p_user"],
            evaluator_client=evaluator_client,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_execute, job): job for job in pending_jobs}
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Static Trials"):
            job_info = future_map[future]
            try:
                res = future.result()
                records.append(res)
                # checkpoint after every 5 completed trials
                if checkpoint_path and len(records) % 5 == 0:
                    pd.DataFrame(records).to_csv(checkpoint_path, index=False)
            except Exception as e:
                print(
                    f"\n  trial failed: scenario={job_info['scenario'].id}, "
                    f"condition={job_info['condition']}, p_user={job_info['p_user']:.1f} — {e}"
                )

    if not records:
        raise RuntimeError("All trials failed. Check your API key and model name.")

    final_df = pd.DataFrame(records)
    if checkpoint_path:
        final_df.to_csv(checkpoint_path, index=False)
    return final_df


def run_closed_loop(
    client,
    assistant_model: str,
    evaluator_model: str,
    scenario: Scenario,
    condition: str = "high",
    turns: int = 6,
    initial_belief_H: float = 5.0,
    alpha: float = 0.4,
    beta: float = 0.5,
    evidence_mode: str = "none",
    evidence_type: str = "none",
    evidence_strength: float = 0.0,
    user_noise_std: float = 0.03,
    seed: Optional[int] = None,
    user_mode: str = "llm",
    user_client=None,
    user_model: Optional[str] = None,
    user_sycophancy_liking: str = "medium",
    evaluator_client=None,
    custom_user_message: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the configured user and chatbot conversation, scoring the speaker after every message.

    Conversational evidence is shown in the dialogue. Controlled users also support
    mathematical evidence applied to their internal state. The evaluator scores the user's
    belief after user messages and the assistant's expressed stance after assistant messages.
    """
    if turns < 1:
        raise ValueError("turns must be at least 1.")
    if not BELIEF_H_MIN <= initial_belief_H <= BELIEF_H_MAX:
        raise ValueError(f"initial_belief_H must be between {BELIEF_H_MIN:g} and {BELIEF_H_MAX:g}.")
    if condition not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown condition '{condition}'. Must be one of: {list(SYSTEM_PROMPTS.keys())}")
    if evidence_mode not in ("none", "mathematical", "conversational"):
        raise ValueError("evidence_mode must be 'none', 'mathematical', or 'conversational'.")

    # support legacy evidence_type arg if caller used old signature
    if evidence_mode == "none" and evidence_type != "none":
        evidence_mode = "conversational"

    if user_mode not in ("llm", "controlled"):
        raise ValueError("user_mode must be llm or controlled.")
    if evidence_mode == "mathematical" and user_mode == "llm":
        raise ValueError("Use conversational evidence with an LLM user.")

    # initialize the simulated user's belief state
    pending_user = None
    if user_mode == "llm":
        user = LLMUser(
            client=user_client if user_client is not None else client,
            model=user_model or assistant_model,
            scenario=scenario,
            initial_H=initial_belief_H,
            sycophancy_liking=user_sycophancy_liking,
            seed=seed,
        )
        if custom_user_message and custom_user_message.strip():
            pending_user = user.seed_initial_message(custom_user_message.strip(), initial_H=initial_belief_H)
        else:
            pending_user = user.start()
    else:
        user = ControlledUser(
            initial_p=initial_belief_H,
            alpha=alpha,
            beta=beta,
            noise_std=user_noise_std,
            seed=seed,
        )
    schedule_type = evidence_type if evidence_type != "none" else "constant"
    evidence_schedule = make_evidence_schedule(turns, schedule_type, evidence_strength)
    system_prompt = SYSTEM_PROMPTS[condition]

    # conversation starts with the system prompt
    conversation_history: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    trajectory = []
    evaluated_initial_H = None
    requested_initial_p1, requested_initial_p2 = belief_probabilities_from_H(initial_belief_H)

    for t in range(turns):
        ext_evidence_strength = evidence_schedule[t] if evidence_type != "none" else 0.0
        controlled_belief_before = user.p if user_mode == "controlled" else None

        # conversational evidence: inject ground truth text into the user's dialogue message
        evidence_text_for_msg = None
        if evidence_mode == "conversational" and ext_evidence_strength > 0:
            evidence_text_for_msg = f"Independent sources report: {scenario.ground_truth}"

        if user_mode == "llm":
            user_turn_msg = pending_user["message"]
            if evidence_text_for_msg:
                user_turn_msg += "\n\nNote: " + evidence_text_for_msg
                # Both agents see the same external evidence in the conversation.
                user.messages.append({"role": "user", "content": evidence_text_for_msg})
        else:
            user_turn_msg = generate_user_message(
                scenario=scenario, p_user=user.p,
                evidence_text=evidence_text_for_msg, user_stance=user.stance,
            )

        conversation_history.append({"role": "user", "content": user_turn_msg})

        user_dialogue_text = "\n\n".join(
            f"{entry['role'].capitalize()}: {entry['content']}"
            for entry in conversation_history if entry["role"] != "system"
        )
        user_eval = evaluate_response(
            client=evaluator_client if evaluator_client is not None else client,
            model=evaluator_model,
            dialogue_text=user_dialogue_text,
            ground_truth=scenario.ground_truth,
            false_belief=scenario.false_belief,
            latest_speaker="user",
        )
        if t == 0:
            initial_eval = user_eval
            evaluated_initial_H = user_eval["H"]

        # get assistant reply in the ongoing conversation
        assistant_turn_reply = call_llm(
            client=client,
            model=assistant_model,
            messages=conversation_history,
            temperature=0.0,
        )

        if not assistant_turn_reply:
            raise RuntimeError(f"Assistant returned empty response at turn {t + 1}.")

        conversation_history.append({"role": "assistant", "content": assistant_turn_reply})

        dialogue_text = "\n\n".join(
            f"{entry['role'].capitalize()}: {entry['content']}"
            for entry in conversation_history if entry["role"] != "system"
        )
        turn_eval = evaluate_response(
            client=evaluator_client if evaluator_client is not None else client,
            model=evaluator_model,
            dialogue_text=dialogue_text,
            ground_truth=scenario.ground_truth,
            false_belief=scenario.false_belief,
            latest_speaker="assistant",
        )

        if user_mode == "llm" and t < turns - 1:
            pending_user = user.observe(assistant_turn_reply)
        elif user_mode == "controlled":
            user.update(
                bot_stance=stance_from_logodds(turn_eval["H"]),
                external_evidence=ext_evidence_strength,
            )

        trajectory.append({
            "turn": t + 1,
            "scenario_id": scenario.id,
            "initial_belief_setting": initial_belief_H,
            "initial_belief_setting_p1": requested_initial_p1,
            "initial_belief_setting_p2": requested_initial_p2,
            "initial_belief_H": evaluated_initial_H,
            "initial_belief_p_false": initial_eval["p_false"],
            "initial_belief_p_truth": initial_eval["p_truth"],
            "initial_belief_p1": initial_eval["p1"],
            "initial_belief_p2": initial_eval["p2"],
            "initial_belief_rationale": initial_eval.get("rationale", ""),
            "user_H": user_eval["H"],
            "user_p1": user_eval["p1"],
            "user_p2": user_eval["p2"],
            "user_p_false": user_eval["p_false"],
            "user_p_truth": user_eval["p_truth"],
            "user_H_rationale": user_eval.get("rationale", ""),
            "condition": condition,
            "user_mode": user_mode,
            "controlled_belief_before": controlled_belief_before,
            "controlled_belief_after": user.p if user_mode == "controlled" else None,
            "user_model": (user_model or assistant_model) if user_mode == "llm" else "",
            "user_sycophancy_liking": user_sycophancy_liking if user_mode == "llm" else "",
            "external_evidence": ext_evidence_strength,
            "evidence_mode": evidence_mode,
            "user_msg": user_turn_msg,
            "assistant_reply": assistant_turn_reply,
            "H": turn_eval["H"],
            "assistant_H": turn_eval["H"],
            "assistant_p1": turn_eval["p1"],
            "assistant_p2": turn_eval["p2"],
            "p_false": turn_eval["p_false"],
            "p_truth": turn_eval["p_truth"],
            "H_rationale": turn_eval.get("rationale", ""),
            "assistant_p_false": turn_eval["p_false"],
            "assistant_p_truth": turn_eval["p_truth"],
            "assistant_H_rationale": turn_eval.get("rationale", ""),
        })

    return pd.DataFrame(trajectory)


def sanity_check_trial(
    client,
    assistant_model: str,
    evaluator_model: str,
    scenario: Optional[Scenario] = None,
    evaluator_client=None,
) -> Dict[str, Any]:
    # run one trial each for low and high conditions to verify the whole pipeline works
    if scenario is None:
        scenario = SCENARIOS[0]

    print(f"  sanity checking on scenario: '{scenario.id}'")
    res_low = run_static_trial(
        client=client,
        assistant_model=assistant_model,
        evaluator_model=evaluator_model,
        scenario=scenario,
        condition="low",
        initial_belief_H=5.0,
        evaluator_client=evaluator_client,
    )
    print(f"  low  → initial H={res_low['initial_belief_H']:.2f}, final H={res_low['H']:.2f}")

    res_high = run_static_trial(
        client=client,
        assistant_model=assistant_model,
        evaluator_model=evaluator_model,
        scenario=scenario,
        condition="high",
        initial_belief_H=5.0,
        evaluator_client=evaluator_client,
    )
    print(f"  high → initial H={res_high['initial_belief_H']:.2f}, final H={res_high['H']:.2f}")

    return {"low": res_low, "high": res_high}
