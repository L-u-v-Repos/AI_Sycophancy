"""Run each scenario × assistant condition × user sycophancy liking once."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sycophancy-mpl")

import pandas as pd
from dotenv import dotenv_values

from sycophancy_testbed.analysis import summarize_trajectory
from sycophancy_testbed.client import get_client
from sycophancy_testbed.config import (
    CONDITIONS,
    DEFAULT_GROQ_MODEL,
    DEFAULT_GEMINI_MODEL,
    SUPPORTED_GROQ_MODELS,
    SUPPORTED_GEMINI_MODELS,
    SCENARIOS,
)
from sycophancy_testbed.experiments import run_closed_loop
from sycophancy_testbed.plotting import plot_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--provider",
        choices=["groq", "gemini"],
        default="groq",
        help="LLM provider: groq or gemini (default: groq)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to evaluate across the benchmark (default: provider default model)",
    )
    args = parser.parse_args()
    if args.model is None:
        args.model = DEFAULT_GEMINI_MODEL if args.provider == "gemini" else DEFAULT_GROQ_MODEL

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    values = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
    clients = {}
    if args.provider == "gemini":
        gemini_key = os.environ.get("GEMINI_API_KEY") or values.get("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("Missing GEMINI_API_KEY in environment or .env")
        for role in ("ASSISTANT", "USER", "EVALUATOR"):
            clients[role] = get_client("gemini", gemini_key)
    else:
        for role in ("ASSISTANT", "USER", "EVALUATOR"):
            key = os.environ.get(role + "_API_KEY") or values.get(role + "_API_KEY")
            if not key:
                raise ValueError(f"Missing {role}_API_KEY")
            clients[role] = get_client("groq", key)
    config = {"provider": args.provider, "model": args.model,
              "turns": 6, "initial_belief_H": 5.0, "evidence": "none", "seed": 42,
              "scenarios": [s.id for s in SCENARIOS], "conditions": CONDITIONS,
              "sycophancy_liking_levels": CONDITIONS, "repetitions": 1}
    manifest = output / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Output directory contains a different experiment configuration.")
    manifest.write_text(json.dumps(config, indent=2))
    summaries = []
    total = len(SCENARIOS) * len(CONDITIONS) ** 2
    for scenario in SCENARIOS:
        for condition in CONDITIONS:
            for liking in CONDITIONS:
                name = f"{scenario.id}__{condition}__{liking}"
                path = output / f"{name}.csv"
                print(f"[{len(summaries) + 1}/{total}] {name}", flush=True)
                if path.exists():
                    df = pd.read_csv(path)
                    print("  Already completed; loading saved conversation.", flush=True)
                else:
                    started = time.monotonic()
                    try:
                        df = run_closed_loop(
                            client=clients["ASSISTANT"], assistant_model=args.model,
                            evaluator_model=args.model,
                            evaluator_client=clients["EVALUATOR"],
                            user_client=clients["USER"], user_model=args.model,
                            scenario=scenario, condition=condition, user_sycophancy_liking=liking,
                            turns=6, initial_belief_H=5.0, seed=42,
                        )
                    except Exception as exc:
                        # Never include credentials in a persisted error.
                        message = str(exc)
                        for role in ("ASSISTANT", "USER", "EVALUATOR"):
                            k = os.environ.get(role + "_API_KEY") or values.get(role + "_API_KEY")
                            if k:
                                message = message.replace(k, "[REDACTED]")
                        gemini_k = os.environ.get("GEMINI_API_KEY") or values.get("GEMINI_API_KEY")
                        if gemini_k:
                            message = message.replace(gemini_k, "[REDACTED]")
                        (output / "error.json").write_text(json.dumps({
                            "conversation": name, "error_type": type(exc).__name__,
                            "message": message,
                        }, indent=2))
                        print(f"Stopped: {type(exc).__name__}: {message}", flush=True)
                        raise SystemExit(1)
                    df["user_provider"] = args.provider
                    temporary = path.with_suffix(".tmp")
                    df.to_csv(temporary, index=False)
                    temporary.replace(path)
                    print(f"  Saved 6 turns in {time.monotonic() - started:.1f}s", flush=True)
                plot_trajectory(df, str(output / f"{name}.png"))
                summaries.append({"scenario_id": scenario.id, "condition": condition,
                                  "user_sycophancy_liking": liking, **summarize_trajectory(df)})
                pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.groupby(["condition", "user_sycophancy_liking"])[
        ["H"]
    ].mean().to_csv(output / "summary_by_condition_and_sycophancy_liking.csv")
    print(f"Completed {total} conversations. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
