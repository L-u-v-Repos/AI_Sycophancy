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
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # plot metric bars
    ax.bar(x - 1.5 * width, summary_df["mean_H"], width, label="Mean H (Alignment - Contradiction)", color="#2b5c8f")
    ax.bar(x - 0.5 * width, summary_df["mean_alignment"], width, label="Mean Alignment", color="#d95f02")
    ax.bar(x + 0.5 * width, summary_df["mean_contradiction"], width, label="Mean Contradiction", color="#7570b3")
    ax.bar(x + 1.5 * width, summary_df["s_wrong"], width, label="P(Reinf. | User Wrong)", color="#e7298a")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in conditions], fontsize=11, fontweight="bold")
    ax.set_ylabel("Score (0.0 to 1.0 or -1.0 to 1.0 for H)", fontsize=10)
    ax.set_title("Sycophancy Metrics by System Prompt Condition", fontsize=12, fontweight="bold")
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

    # prefer user_stance (signed) column; fall back to user_conviction
    cond_x_col = "user_stance" if "user_stance" in df.columns else "user_conviction"
    x_label = "User Stance in False Belief (-1.0 = rejects, +1.0 = endorses)"

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
                x_vals = np.linspace(-1.0, 1.0, 50)
                # correct OLS line: y = chi * x + intercept (uses stored intercept, not point-slope hack)
                y_vals = np.clip(chi * x_vals + intercept, -1.0, 1.0)
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
    ax.set_ylabel("Assistant Alignment Score H (-1.0 to +1.0)", fontsize=11)
    ax.set_title("Response Surface: Susceptibility Slope χ Across Conditions", fontsize=12, fontweight="bold")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_trajectory(trajectory_df: pd.DataFrame, output_path: str) -> None:
    # ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, dpi=150)

    turns = trajectory_df["turn"]

    # top plot: belief probability
    ax1.plot(turns, trajectory_df["user_belief_prob"], marker="o", color="#d95f02", linewidth=2.2, label="P(User False Belief)")
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax1.set_ylabel("Belief Probability", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Closed-Loop Belief Dynamics & Assistant Sycophancy", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # bottom plot: assistant alignment score h
    ax2.plot(turns, trajectory_df["assistant_H"], marker="s", color="#2b5c8f", linewidth=2.0, label="Assistant H")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Turn Number", fontsize=11)
    ax2.set_ylabel("Assistant H", fontsize=11)
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
