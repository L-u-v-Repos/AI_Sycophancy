import math
import os
import matplotlib
# set non-interactive backend for headless execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_sycophancy_bars(summary_df: pd.DataFrame, output_path: str) -> None:
    # ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    conditions = summary_df["condition"].tolist()
    x = np.arange(len(conditions))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    ax.bar(x - width / 2, summary_df["mean_initial_belief_H"], width, label="Initial belief H", color="#8da0cb")
    ax.bar(x + width / 2, summary_df["mean_H"], width, label="Final H", color="#2b5c8f")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in conditions], fontsize=11, fontweight="bold")
    ax.set_ylabel("H (−5 to +5)", fontsize=10)
    ax.set_ylim(-5.25, 5.25)
    ax.set_title("Initial User Belief and Assistant Stance by Condition", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_response_surface(
    df: pd.DataFrame,
    slopes_df: pd.DataFrame,
    output_path: str,
) -> None:
    # ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    colors = {"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728"}
    condition_order = ["low", "medium", "high"]

    cond_x_col = "initial_belief_H"
    x_label = "Initial belief H (-5 = near-certain ground truth, +5 = near-certain misconception)"

    # map slopes dataframe for quick lookup
    slopes_map = {}
    if slopes_df is not None and not slopes_df.empty:
        for _, row in slopes_df.iterrows():
            slopes_map[row["condition"]] = (row["slope_chi"], row["intercept"], row["r_squared"])

    for cond in condition_order:
        subset = df[df["condition"] == cond]
        if subset.empty:
            continue

        c = colors.get(cond, "gray")
        # jitter points slightly for visual clarity
        jitter = np.random.normal(0, 0.015, size=len(subset))
        ax.scatter(
            subset[cond_x_col] + jitter,
            subset["H"],
            color=c,
            alpha=0.6,
            s=45,
            label=f"{cond.capitalize()} data",
        )

        # plot fitted regression line only when slope is valid
        if cond in slopes_map:
            chi, intercept, r2 = slopes_map[cond]
            if not (math.isnan(chi) or math.isnan(r2)):
                x_vals = np.linspace(-5.0, 5.0, 50)
                # correct OLS line: y = chi * x + intercept (uses stored intercept, not point-slope hack)
                y_vals = np.clip(chi * x_vals + intercept, -5.0, 5.0)
                ax.plot(
                    x_vals,
                    y_vals,
                    color=c,
                    linewidth=2.2,
                    label=f"{cond.capitalize()} fit (χ={chi:.2f}, R²={r2:.2f})",
                )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Final evaluator H (−5 to +5)", fontsize=11)
    ax.set_title("Response Surface: Susceptibility Slope χ Across Conditions", fontsize=12, fontweight="bold")
    ax.set_xlim(-5.25, 5.25)
    ax.set_ylim(-5.25, 5.25)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_trajectory(trajectory_df: pd.DataFrame, output_path: str) -> None:
    # Show user belief after each user message and assistant expressed stance after each reply.
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)
    turns = trajectory_df["turn"].tolist()
    x_values = list(range(1, len(turns) * 2 + 1))
    h_values = []
    labels = []
    for turn_number, (_, row) in zip(turns, trajectory_df.iterrows()):
        h_values.extend([float(row["user_H"]), float(row["assistant_H"])])
        labels.extend([f"User {turn_number}", f"Assistant {turn_number}"])
    ax.plot(x_values, h_values, marker="o", color="#2b5c8f", linewidth=2.2)
    ax.set_xticks(x_values, labels, rotation=35, ha="right")
    ax.set_ylabel("H (−5 to +5)", fontsize=11)
    ax.set_ylim(-5.25, 5.25)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("User Belief and Assistant Expressed Stance", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
