from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from typing import Any, Dict, List, Optional
import pandas as pd
from tqdm import tqdm

from .client import call_llm, call_llm_with_usage
from .config import (
    Scenario,
    SYSTEM_PROMPTS,
    SCENARIOS,
    DEFAULT_P_USER_GRID,
    CONDITIONS,
)
from .evaluator import evaluate_response
from .metrics import conviction_from_logodds, short_hash, stance_from_logodds
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
    p_user: float,
    assistant_temperature: float = 0.0,
    evaluator_client=None,
    custom_user_message: Optional[str] = None,
) -> Dict[str, Any]:
    if condition not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown condition '{condition}'. Must be one of: {list(SYSTEM_PROMPTS.keys())}")

    user_stance_val = stance_from_logodds(p_user)
    user_conviction_val = conviction_from_logodds(p_user)

    system_prompt = SYSTEM_PROMPTS[condition]
    if custom_user_message and custom_user_message.strip():
        user_msg = custom_user_message.strip()
    else:
        user_msg = generate_user_message(scenario, p_user=p_user, user_stance=user_stance_val)

    # record version hashes for reproducibility
    scenario_hash = short_hash(scenario)
    sysprompt_hash = short_hash(system_prompt)
    userprompt_hash = short_hash(user_msg)

    # get assistant response with token usage
    asst_result = call_llm_with_usage(
        client=client,
        model=assistant_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=assistant_temperature,
    )
    assistant_response = asst_result["content"]
    asst_usage = asst_result["usage"]

    if not assistant_response:
        raise RuntimeError(
            f"Assistant returned an empty response for scenario={scenario.id}, condition={condition}"
        )

    # score the assistant response with the evaluator judge
    eval_result = evaluate_response(
        client=evaluator_client if evaluator_client is not None else client,
        model=evaluator_model,
        scenario=scenario,
        user_msg=user_msg,
        assistant_response=assistant_response,
    )

    return {
        "scenario_id": scenario.id,
        "domain": scenario.domain,
        "condition": condition,
        "p_user": p_user,
        # user_stance: signed [-1, 1] — positive means user endorses false belief
        "user_stance": user_stance_val,
        # user_conviction: unsigned [0, 1] — certainty regardless of direction
        "user_conviction": user_conviction_val,
        "user_msg": user_msg,
        "assistant_response": assistant_response,
        # token usage
        "bot_prompt_tokens": asst_usage.get("prompt_tokens"),
        "bot_completion_tokens": asst_usage.get("completion_tokens"),
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
            records = existing_df.to_dict(orient="records")
            for r in records:
                key = (r["scenario_id"], r["condition"], round(float(r["p_user"]), 4))
                completed_keys.add(key)
            print(f"  resuming from checkpoint: {len(records)} trials already done")
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
            p_user=job["p_user"],
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
    initial_p: float = 2.0,
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
    user_trust: str = "medium",
    evaluator_client=None,
    custom_user_message: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run a closed-loop multi-turn belief drift simulation.

    evidence_mode controls how external truth evidence is applied:
      - 'none': no evidence
      - 'mathematical': evidence updates user belief state directly (delta_p -= beta * E)
        but is NOT injected into the conversation text
      - 'conversational': evidence is injected as text into the dialogue
        but does NOT directly update the belief state numerically
    Keeping these modes separate prevents double-counting the same evidence.
    """
    if turns < 1:
        raise ValueError("turns must be at least 1.")
    if condition not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown condition '{condition}'. Must be one of: {list(SYSTEM_PROMPTS.keys())}")
    if evidence_mode not in ("none", "mathematical", "conversational"):
        raise ValueError("evidence_mode must be 'none', 'mathematical', or 'conversational'.")

    # support legacy evidence_type arg if caller used old signature
    if evidence_mode == "none" and evidence_type != "none":
        evidence_mode = "conversational"

    if user_mode not in ("llm", "controlled"):
        raise ValueError("user_mode must be llm or controlled.")
    if user_mode == "llm" and evidence_mode == "mathematical":
        raise ValueError("LLM users require conversational evidence; use --user-mode controlled for mathematical evidence.")

    # initialize the simulated user's belief state
    pending_user = None
    if user_mode == "llm":
        user = LLMUser(
            client=user_client if user_client is not None else client,
            model=user_model or assistant_model,
            scenario=scenario,
            initial_p=initial_p,
            trust=user_trust,
            seed=seed,
        )
        if custom_user_message and custom_user_message.strip():
            pending_user = user.seed_initial_message(custom_user_message.strip(), initial_p=initial_p)
        else:
            pending_user = user.start()
    else:
        user = ControlledUser(
            initial_p=initial_p,
            alpha=alpha,
            beta=beta,
            noise_std=user_noise_std,
            seed=seed,
        )
    evidence_schedule = make_evidence_schedule(turns, evidence_type if evidence_type != "none" else "constant", evidence_strength)
    system_prompt = SYSTEM_PROMPTS[condition]

    # conversation starts with the system prompt
    conversation_history: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    trajectory = []

    for t in range(turns):
        current_p = user.p
        current_stance = user.stance
        current_conviction = user.conviction
        current_prob = user.belief_probability
        ext_evidence_strength = evidence_schedule[t] if evidence_type != "none" else 0.0

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
                scenario=scenario, p_user=current_p,
                evidence_text=evidence_text_for_msg, user_stance=current_stance,
            )

        conversation_history.append({"role": "user", "content": user_turn_msg})

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

        # evaluate how sycophantic this turn's reply was
        eval_turn = evaluate_response(
            client=evaluator_client if evaluator_client is not None else client,
            model=evaluator_model,
            scenario=scenario,
            user_msg=user_turn_msg,
            assistant_response=assistant_turn_reply,
        )

        # update user belief:
        # bot_stance is the evaluator's H score, used as a proxy for the bot's expressed position
        # (+1 = bot fully endorsed false belief, -1 = bot fully contradicted it)
        # mathematical evidence: only subtracts from belief state, NOT added to dialogue text
        math_evidence = ext_evidence_strength if evidence_mode == "mathematical" else 0.0
        belief_reason = ""
        if user_mode == "llm":
            pending_user = user.observe(assistant_turn_reply)
            belief_reason = pending_user["reason"]
        else:
            user.update(bot_stance=eval_turn["H"], external_evidence=math_evidence)

        trajectory.append({
            "turn": t + 1,
            "scenario_id": scenario.id,
            "initial_belief": scenario.false_belief,
            "initial_p": initial_p,
            "condition": condition,
            "user_mode": user_mode,
            "user_model": (user_model or assistant_model) if user_mode == "llm" else "",
            "user_trust": user_trust if user_mode == "llm" else "",
            "user_p_after": user.p,
            "user_belief_prob_after": user.belief_probability,
            "user_belief_reason": belief_reason,
            "user_reaction": pending_user["message"] if pending_user else "",
            "user_p": current_p,
            "user_stance": current_stance,
            "user_conviction": current_conviction,
            "user_belief_prob": current_prob,
            "external_evidence": ext_evidence_strength,
            "evidence_mode": evidence_mode,
            "assistant_H": eval_turn["H"],
            "alignment": eval_turn["alignment_with_false_belief"],
            "contradiction": eval_turn["contradiction_of_false_belief"],
            "accuracy": eval_turn["accuracy_score"],
            "reasoning": eval_turn["reasoning_quality"],
            "user_msg": user_turn_msg,
            "assistant_reply": assistant_turn_reply,
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
        p_user=2.0,
        evaluator_client=evaluator_client,
    )
    print(f"  low  → H={res_low['H']:.2f}, alignment={res_low['alignment_with_false_belief']:.2f}")

    res_high = run_static_trial(
        client=client,
        assistant_model=assistant_model,
        evaluator_model=evaluator_model,
        scenario=scenario,
        condition="high",
        p_user=2.0,
        evaluator_client=evaluator_client,
    )
    print(f"  high → H={res_high['H']:.2f}, alignment={res_high['alignment_with_false_belief']:.2f}")

    return {"low": res_low, "high": res_high}
