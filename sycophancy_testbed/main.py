import argparse
import os
import sys
import time

from dotenv import load_dotenv

from .analysis import (
    compute_response_slopes,
    compute_response_surface,
    summarize_static,
    summarize_trajectory,
)
from .client import get_client
from .config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_OPENROUTER_MODEL,
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


def parse_args() -> argparse.Namespace:
    # parse command line options
    parser = argparse.ArgumentParser(
        description="Sycophancy & Closed-Loop Belief Drift Testbed"
    )
    parser.add_argument(
        "--provider",
        choices=["groq", "openrouter"],
        default="groq",
        help="LLM provider: groq (recommended free tier) or openrouter (default: groq)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Chatbot API key override (defaults to ASSISTANT_API_KEY in .env)",
    )
    parser.add_argument(
        "--assistant-model",
        type=str,
        default=None,
        help="Model ID for the assistant being evaluated",
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
            "How external evidence is applied in closed-loop: "
            "'conversational' injects it as dialogue text, "
            "'mathematical' applies it directly to belief state only (default: conversational)"
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
    parser.add_argument("--user-provider", choices=["groq", "openrouter"], default=None,
                        help="User bot provider (defaults to assistant provider)")
    parser.add_argument("--user-api-key", default=None,
                        help="User bot key override; prefer USER_API_KEY in .env")
    parser.add_argument("--user-model", default=None, help="User bot model ID")
    parser.add_argument("--user-trust", choices=["low", "medium", "high", "all"], default="medium",
                        help="User trust in chatbot; all runs three independent conversations")
    parser.add_argument("--initial-p", type=float, default=2.0,
                        help="Initial log-odds of scenario belief (default 2, about 88%% probability)")
    parser.add_argument("--condition", choices=["low", "medium", "high"], default="high",
                        help="Assistant sycophancy condition for closed-loop runs")
    return parser.parse_args()


def main() -> None:
    # load .env from the repo root so api keys are available
    load_dotenv()

    args = parse_args()

    default_model = DEFAULT_GROQ_MODEL if args.provider == "groq" else DEFAULT_OPENROUTER_MODEL
    api_key = args.api_key or os.environ.get("ASSISTANT_API_KEY")
    evaluator_key = args.evaluator_api_key or os.environ.get("EVALUATOR_API_KEY")
    user_key = args.user_api_key or os.environ.get("USER_API_KEY")
    if args.run != "none":
        required_keys = {"ASSISTANT_API_KEY": api_key, "EVALUATOR_API_KEY": evaluator_key}
        if args.run == "closed_loop" and args.user_mode == "llm":
            required_keys["USER_API_KEY"] = user_key
        missing = [name for name, key in required_keys.items() if not key or not key.strip()]
        if missing:
            print("Error: set separate role keys in .env: " + ", ".join(missing))
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
        print("\nSanity check passed.")
        print(f"  low  note: {results['low'].get('notes')}")
        print(f"  high note: {results['high'].get('notes')}")

    # --- run: pilot ---
    elif args.run == "pilot":
        pilot_scs = PILOT_SCENARIOS if not args.scenario else scenario_list
        n_trials = len(pilot_scs) * 3 * 5
        print(f"\n--- Pilot Experiment ({len(pilot_scs)} scenarios × 3 conditions × 5 beliefs = {n_trials} trials) ---")
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
        n_trials = len(scenario_list) * 3 * 5
        print(f"\n--- Full Experiment ({len(scenario_list)} scenarios × 3 conditions × 5 beliefs = {n_trials} trials) ---")
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
                DEFAULT_GROQ_MODEL if user_provider == "groq" else DEFAULT_OPENROUTER_MODEL)
            print(f"User bot: {user_provider} / {user_model}; trust={args.user_trust}")
        elif args.user_trust != "medium":
            raise ValueError("--user-trust applies only to --user-mode llm.")

        trust_levels = ["low", "medium", "high"] if args.user_trust == "all" else [args.user_trust]
        summaries = []
        for trust in trust_levels:
            traj_df = run_closed_loop(
                client=client,
                assistant_model=assistant_model,
                evaluator_model=evaluator_model,
                evaluator_client=evaluator_client,
                scenario=scenario_list[0],
                condition=args.condition,
                turns=args.turns,
                initial_p=args.initial_p,
                evidence_type=args.evidence,
                evidence_mode=evidence_mode,
                evidence_strength=evidence_strength,
                seed=args.seed,
                user_mode=args.user_mode,
                user_client=user_client,
                user_model=user_model,
                user_trust=trust,
            )
            traj_df["user_provider"] = user_provider if args.user_mode == "llm" else ""
            traj_summary = summarize_trajectory(traj_df)
            summaries.append({"user_trust": trust if args.user_mode == "llm" else "",
                              "condition": args.condition, **traj_summary})
            print(f"\nTrajectory Summary ({trust} trust):")
            for k, v in traj_summary.items():
                print(f"  {k}: {v}")

            suffix = f"_{trust}" if args.user_trust == "all" else ""
            traj_csv = os.path.join(args.output_dir, f"closed_loop_trajectory{suffix}.csv")
            traj_plot = os.path.join(args.output_dir, f"closed_loop_plot{suffix}.png")
            traj_df.to_csv(traj_csv, index=False)
            plot_trajectory(traj_df, traj_plot)
            print(f"\nTrajectory saved: {traj_csv}")
            print(f"Plot saved:       {traj_plot}")
        import pandas as pd
        pd.DataFrame(summaries).to_csv(
            os.path.join(args.output_dir, "closed_loop_summary.csv"), index=False)



if __name__ == "__main__":
    main()
