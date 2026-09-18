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
        help="API key override (defaults to GROQ_API_KEY or OPENROUTER_API_KEY in .env)",
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
    return parser.parse_args()


def main() -> None:
    # load .env from the repo root so api keys are available
    load_dotenv()

    args = parse_args()

    # resolve api key and default model for the chosen provider
    if args.provider == "groq":
        api_key = args.api_key or os.environ.get("GROQ_API_KEY")
        default_model = DEFAULT_GROQ_MODEL
        env_var_name = "GROQ_API_KEY"
    else:
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
        default_model = DEFAULT_OPENROUTER_MODEL
        env_var_name = "OPENROUTER_API_KEY"

    # check api key before doing anything else
    if args.run != "none" and not api_key:
        print(f"Error: no API key found for provider '{args.provider}'.")
        print(f"Set {env_var_name} in your .env file or pass --api-key <key>.")
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
        traj_df = run_closed_loop(
            client=client,
            assistant_model=assistant_model,
            evaluator_model=evaluator_model,
            scenario=scenario_list[0],
            condition="high",
            turns=args.turns,
            evidence_type=args.evidence,
            evidence_mode=evidence_mode,
            evidence_strength=evidence_strength,
            seed=args.seed,
        )

        traj_summary = summarize_trajectory(traj_df)
        print("\nTrajectory Summary:")
        for k, v in traj_summary.items():
            print(f"  {k}: {v}")

        traj_csv = os.path.join(args.output_dir, "closed_loop_trajectory.csv")
        traj_plot = os.path.join(args.output_dir, "closed_loop_plot.png")
        traj_df.to_csv(traj_csv, index=False)
        plot_trajectory(traj_df, traj_plot)
        print(f"\nTrajectory saved: {traj_csv}")
        print(f"Plot saved:       {traj_plot}")


if __name__ == "__main__":
    main()
