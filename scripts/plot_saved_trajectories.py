"""Generate separate user_H and assistant_H charts from saved trajectory CSVs."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sycophancy-mpl")

import pandas as pd
from sycophancy_testbed.plotting import plot_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="results", help="CSV file or directory to search recursively")
    args = parser.parse_args()
    source = Path(args.path)
    if not source.exists():
        parser.error(f"Path does not exist: {source}")
    files = [source] if source.is_file() else sorted(source.rglob("*.csv"))
    plotted = 0
    for csv in files:
        columns = pd.read_csv(csv, nrows=0).columns
        if not {"turn", "user_H", "assistant_H"}.issubset(columns):
            continue
        df = pd.read_csv(csv)
        if df.empty:
            continue
        stem = csv.stem.replace("_trajectory", "_plot") if "_trajectory" in csv.stem else csv.stem
        user, assistant = plot_trajectory(df, str(csv.with_name(stem + ".png")))
        print(f"{user}\n{assistant}", flush=True)
        plotted += 1
    print(f"Generated {plotted * 2} charts from {plotted} trajectories.")


if __name__ == "__main__":
    main()
