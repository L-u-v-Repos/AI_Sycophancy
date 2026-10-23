import argparse
import os
import sys
import time
import pandas as pd

from dotenv import load_dotenv

from .analysis import (
    compute_response_slopes,
    compute_response_surface,
    summarize_static,
    summarize_trajectory,
)
from .client import get_client
from .config import (
    DEFAULT_P_USER_GRID,
    DEFAULT_GROQ_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_GEMINI_MODEL,
    SUPPORTED_GROQ_MODELS,
    SUPPORTED_GEMINI_MODELS,
    CONDITIONS,
    PILOT_SCENARIOS,
    SCENARIOS,
)
from .experiments import (
    run_closed_loop,
    run_static_experiment,
    sanity_check_trial,
)
from .plotting import (
    plot_response_surface,
    plot_sycophancy_bars,
    plot_trajectory,
)


def print_interaction_card(title: str, data: dict) -> None:
    user_msg = data.get("user_message") or data.get("user_msg", "")
    assistant_resp = data.get("assistant_response") or data.get("assistant_reply", "")
    h_score = float(data.get("H", 0.0))

    border = "═" * 70
    print("\n" + border)
    print(f"  INTERACTION CARD: {str(title).upper()}")
    print(border)
    print("[USER PROMPT]")
    for line in str(user_msg).strip().splitlines():
        print(f"   {line}")
    print("\n[ASSISTANT RESPONSE]")
    for line in str(assistant_resp).strip().splitlines():
        print(f"   {line}")
    print("\n[EVALUATOR GRADE]")
    user_h = data.get("user_H", data.get("initial_belief_H"))
    if user_h is not None:
        print(f"   After user message:     {float(user_h):+.2f}")
        user_rationale = data.get("user_H_rationale", data.get("initial_belief_rationale", ""))
        if user_rationale:
            print(f"   User rationale:         {user_rationale}")
    assistant_h = data.get("assistant_H", h_score)
    print(f"   Assistant expressed stance: {float(assistant_h):+.2f}")
    assistant_rationale = data.get("assistant_H_rationale", data.get("H_rationale", data.get("rationale", "")))
    if assistant_rationale:
        print(f"   Assistant stance rationale: {assistant_rationale}")
    print(border + "\n")


def parse_args() -> argparse.Namespace:
    # parse command line options
    parser = argparse.ArgumentParser(
        description="Sycophancy & Closed-Loop Belief Drift Testbed"
    )
    parser.add_argument(
        "--provider",
        choices=["groq", "openrouter", "gemini"],
        default="groq",
        help="LLM provider: groq (recommended free tier), openrouter, or gemini (default: groq)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Chatbot API key override (defaults to ASSISTANT_API_KEY in .env)",
    )
    parser.add_argument(
        "--model",
        "--assistant-model",
        dest="assistant_model",
        type=str,
        default=None,
        help=f"Model ID for the assistant being evaluated. High-rate-limit models on Groq: {', '.join(SUPPORTED_GROQ_MODELS)} (default: {DEFAULT_GROQ_MODEL})",
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        default=None,
        help="Model ID for the impartial evaluator judge",
    )
    parser.add_argument(
        "--run",
        choices=["sanity", "pilot", "closed_loop", "full", "none"],
        default="sanity",
        help="Experiment workflow to run (default: sanity)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=time.strftime("./results/run_%Y%m%d_%H%M%S"),
        help="Directory to write CSV results and plots",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of concurrent API workers (default: 2)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=6,
        help="Number of dialogue turns for closed-loop simulation (default: 6)",
    )
    parser.add_argument(
        "--evidence",
        choices=["none", "constant", "step", "ramp"],
        default="none",
        help="External truth evidence schedule for closed-loop run (default: none)",
    )
    parser.add_argument(
        "--evidence-mode",
        choices=["mathematical", "conversational"],
        default="conversational",
        help=(
            "Evidence mode for closed-loop. Evidence is provided in the dialogue so the user "
            "and chatbot can consider it (default: conversational)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for closed-loop noise reproducibility (default: none)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Restrict run to a specific scenario by ID (e.g. quantum_superposition)",
    )
    parser.add_argument("--evaluator-api-key", default=None,
                        help="Judge key override; defaults to EVALUATOR_API_KEY in .env")
    parser.add_argument("--user-mode", choices=["llm", "controlled"], default="llm",
                        help="Closed-loop user: LLM bot (default) or legacy numerical simulation")
    parser.add_argument("--user-provider", choices=["groq", "openrouter", "gemini"], default=None,
                        help="User bot provider (defaults to assistant provider)")
    parser.add_argument("--user-api-key", default=None,
                        help="User bot key override; prefer USER_API_KEY in .env")
    parser.add_argument("--user-model", default=None, help="User bot model ID")
    parser.add_argument("--user-sycophancy-liking", choices=["low", "medium", "high", "all"], default="medium",
                        help="User liking for sycophantic agreement; all runs three independent conversations")
    parser.add_argument("--initial-belief", "--initial-p", dest="initial_belief_H", type=float, default=5.0,
                        help="Opening belief H from -5 (near-certain ground-truth belief) to +5 (near-certain misconception; default +5)")
    parser.add_argument("--condition", choices=["low", "medium", "high"], default="high",
                        help="Assistant sycophancy condition for closed-loop runs")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print the full dialogue exchanges and final H in terminal")
    return parser.parse_args()


def main() -> None:
    # load .env from the repo root so api keys are available
    load_dotenv()

    args = parse_args()

    if args.provider == "groq":
        default_model = DEFAULT_GROQ_MODEL
    elif args.provider == "gemini":
        default_model = DEFAULT_GEMINI_MODEL
    else:
        default_model = DEFAULT_OPENROUTER_MODEL

    if args.provider == "gemini":
        gemini_env_key = os.environ.get("GEMINI_API_KEY")
        api_key = args.api_key or gemini_env_key or os.environ.get("ASSISTANT_API_KEY")
        evaluator_key = args.evaluator_api_key or gemini_env_key or os.environ.get("EVALUATOR_API_KEY")
        user_key = args.user_api_key or gemini_env_key or os.environ.get("USER_API_KEY")
    else:
        api_key = args.api_key or os.environ.get("ASSISTANT_API_KEY")
        evaluator_key = args.evaluator_api_key or os.environ.get("EVALUATOR_API_KEY")
        user_key = args.user_api_key or os.environ.get("USER_API_KEY")

    if args.run != "none":
        required_keys = {}
        if not api_key:
            required_keys["GEMINI_API_KEY" if args.provider == "gemini" else "ASSISTANT_API_KEY"] = api_key
        if not evaluator_key:
            required_keys["GEMINI_API_KEY" if args.provider == "gemini" else "EVALUATOR_API_KEY"] = evaluator_key
        if args.run == "closed_loop" and args.user_mode == "llm":
            if not user_key:
                u_prov = args.user_provider or args.provider
                required_keys["GEMINI_API_KEY" if u_prov == "gemini" else "USER_API_KEY"] = user_key
        if required_keys:
            print("Error: missing required API keys: " + ", ".join(required_keys.keys()))
            sys.exit(1)

    assistant_model = args.assistant_model or default_model
    evaluator_model = args.evaluator_model or default_model

    print("=" * 60)
    print("Sycophancy & Belief Drift Testbed")
    print("=" * 60)
    print(f"Provider:         {args.provider}")
    print(f"Assistant Model:  {assistant_model}")
    print(f"Evaluator Model:  {evaluator_model}")
    print(f"Run Mode:         {args.run}")
    print(f"Output Dir:       {args.output_dir}")
    print("=" * 60)

    if args.run == "none":
        print("Configuration OK. Exiting without running trials.")
        return

    # initialize api client
    client = get_client(provider=args.provider, api_key=api_key)
    evaluator_client = get_client(provider=args.provider, api_key=evaluator_key)
    os.makedirs(args.output_dir, exist_ok=True)

    # resolve scenario list
    scenario_list = SCENARIOS
    if args.scenario:
        scenario_list = [s for s in SCENARIOS if s.id == args.scenario]
        if not scenario_list:
            print(f"Error: no scenario with id '{args.scenario}'. Available scenarios:")
            for s in SCENARIOS:
                print(f"  {s.id}")
            sys.exit(1)

    # --- run: sanity ---
    if args.run == "sanity":
        print("\n--- Sanity Check (1 scenario, low vs high condition) ---")
        results = sanity_check_trial(
            client=client,
            assistant_model=assistant_model,
            evaluator_model=evaluator_model,
            evaluator_client=evaluator_client,
            scenario=scenario_list[0],
        )

        # Save sanity trials so output directory is never empty
        sanity_df = pd.DataFrame([results["low"], results["high"]])
        sanity_csv = os.path.join(args.output_dir, "sanity_trials.csv")
        sanity_df.to_csv(sanity_csv, index=False)

        # Print live formatted cards so user can inspect actual dialogue and grades
        for cond in ["low", "high"]:
            print_interaction_card(f"Sanity Trial ({cond} condition)", results[cond])

        print(f"Sanity check passed.")
        print(f"Full trial logs & evaluations saved to: {sanity_csv}")

    # --- run: pilot ---
    elif args.run == "pilot":
        pilot_scs = PILOT_SCENARIOS if not args.scenario else scenario_list
        n_trials = len(pilot_scs) * len(CONDITIONS) * len(DEFAULT_P_USER_GRID)
        print(f"\n--- Pilot Experiment ({len(pilot_scs)} scenarios × {len(CONDITIONS)} conditions × {len(DEFAULT_P_USER_GRID)} H settings = {n_trials} trials) ---")
        checkpoint_file = os.path.join(args.output_dir, "pilot_trials.csv")

        df = run_static_experiment(
            client=client,
            assistant_model=assistant_model,
            evaluator_model=evaluator_model,
            evaluator_client=evaluator_client,
            scenarios=pilot_scs,
            max_workers=args.workers,
            checkpoint_path=checkpoint_file,
        )

        summary = summarize_static(df)
        slopes = compute_response_slopes(df)

        print("\nSummary by Condition:")
        print(summary.to_string(index=False))
        print("\nSusceptibility Slopes (χ):")
        print(slopes.to_string(index=False))

        summary.to_csv(os.path.join(args.output_dir, "pilot_summary.csv"), index=False)
        slopes.to_csv(os.path.join(args.output_dir, "pilot_slopes.csv"), index=False)
        surface = compute_response_surface(df)
        surface.to_csv(os.path.join(args.output_dir, "pilot_response_surface.csv"), index=False)
        plot_sycophancy_bars(summary, os.path.join(args.output_dir, "pilot_bars.png"))
        plot_response_surface(df, slopes, os.path.join(args.output_dir, "pilot_response_surface.png"))
        print(f"\nResults saved in: {args.output_dir}")

    # --- run: full ---
    elif args.run == "full":
        n_trials = len(scenario_list) * len(CONDITIONS) * len(DEFAULT_P_USER_GRID)
        print(f"\n--- Full Experiment ({len(scenario_list)} scenarios × {len(CONDITIONS)} conditions × {len(DEFAULT_P_USER_GRID)} H settings = {n_trials} trials) ---")
        checkpoint_file = os.path.join(args.output_dir, "full_trials.csv")

        df = run_static_experiment(
            client=client,
            assistant_model=assistant_model,
            evaluator_model=evaluator_model,
            evaluator_client=evaluator_client,
            scenarios=scenario_list,
            max_workers=args.workers,
            checkpoint_path=checkpoint_file,
        )

        summary = summarize_static(df)
        slopes = compute_response_slopes(df)

        summary.to_csv(os.path.join(args.output_dir, "full_summary.csv"), index=False)
        slopes.to_csv(os.path.join(args.output_dir, "full_slopes.csv"), index=False)
        surface = compute_response_surface(df)
        surface.to_csv(os.path.join(args.output_dir, "full_response_surface.csv"), index=False)
        plot_sycophancy_bars(summary, os.path.join(args.output_dir, "full_bars.png"))
        plot_response_surface(df, slopes, os.path.join(args.output_dir, "full_response_surface.png"))
        print(f"\nResults saved in: {args.output_dir}")

    # --- run: closed_loop ---
    elif args.run == "closed_loop":
        print(f"\n--- Closed-Loop Simulation ({args.turns} turns, evidence={args.evidence}) ---")
        evidence_strength = 0.6 if args.evidence != "none" else 0.0
        evidence_mode = args.evidence_mode if args.evidence != "none" else "none"
        user_client = None
        user_model = None
        if args.user_mode == "llm":
            if evidence_mode == "mathematical":
                raise ValueError("Use conversational evidence with an LLM user.")
            user_provider = args.user_provider or args.provider
            user_client = get_client(provider=user_provider, api_key=user_key)
            user_model = args.user_model or (
                assistant_model if user_provider == args.provider else
                DEFAULT_GROQ_MODEL if user_provider == "groq" else
                DEFAULT_GEMINI_MODEL if user_provider == "gemini" else
                DEFAULT_OPENROUTER_MODEL)
            print(f"User bot: {user_provider} / {user_model}; sycophancy_liking={args.user_sycophancy_liking}")
        elif args.user_sycophancy_liking != "medium":
            raise ValueError("--user-sycophancy-liking applies only to --user-mode llm.")

        liking_levels = ["low", "medium", "high"] if args.user_sycophancy_liking == "all" else [args.user_sycophancy_liking]
        summaries = []
        for liking in liking_levels:
            traj_df = run_closed_loop(
                client=client,
                assistant_model=assistant_model,
                evaluator_model=evaluator_model,
                evaluator_client=evaluator_client,
                scenario=scenario_list[0],
                condition=args.condition,
                turns=args.turns,
                initial_belief_H=args.initial_belief_H,
                evidence_type=args.evidence,
                evidence_mode=evidence_mode,
                evidence_strength=evidence_strength,
                seed=args.seed,
                user_mode=args.user_mode,
                user_client=user_client,
                user_model=user_model,
                user_sycophancy_liking=liking,
            )
            traj_df["user_provider"] = user_provider if args.user_mode == "llm" else ""
            traj_summary = summarize_trajectory(traj_df)
            summaries.append({"user_sycophancy_liking": liking if args.user_mode == "llm" else "",
                              "condition": args.condition, **traj_summary})
            print(f"\nTrajectory Summary ({liking} sycophancy liking):")
            for k, v in traj_summary.items():
                print(f"  {k}: {v}")

            if args.verbose:
                print(f"\n--- Trajectory Exchanges ({liking} sycophancy liking) ---")
                for _, row in traj_df.iterrows():
                    print_interaction_card(f"Turn {int(row['turn'])} ({liking} sycophancy liking)", row.to_dict())

            suffix = f"_{liking}" if args.user_sycophancy_liking == "all" else ""
            traj_csv = os.path.join(args.output_dir, f"closed_loop_trajectory{suffix}.csv")
            traj_plot = os.path.join(args.output_dir, f"closed_loop_plot{suffix}.png")
            traj_df.to_csv(traj_csv, index=False)
            plot_trajectory(traj_df, traj_plot)
            print(f"\nTrajectory saved: {traj_csv}")
            print(f"Plot saved:       {traj_plot}")
        pd.DataFrame(summaries).to_csv(
            os.path.join(args.output_dir, "closed_loop_summary.csv"), index=False)



if __name__ == "__main__":
    main()
