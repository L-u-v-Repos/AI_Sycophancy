from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .metrics import belief_probabilities_from_H


def summarize_static(df: pd.DataFrame) -> pd.DataFrame:
    # aggregate initial and evaluator-assigned final belief across conditions
    required = {"condition", "initial_belief_H", "H"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing expected columns: {missing}")

    records = []
    for cond, grp in df.groupby("condition"):
        n = len(grp)

        records.append({
            "condition": cond,
            "trials": n,
            "mean_initial_belief_H": float(grp["initial_belief_H"].mean()),
            "mean_H": float(grp["H"].mean()),
            "std_H": float(grp["H"].std()) if n > 1 else 0.0,
        })

    return pd.DataFrame(records)


def compute_response_slopes(df: pd.DataFrame) -> pd.DataFrame:
    # calculate the susceptibility slope chi using linear regression per condition
    # regression is run on user_stance (signed [-1,1]); falls back to user_conviction if absent
    if "H" not in df.columns:
        raise ValueError("DataFrame must contain 'H' column.")

    x_col = "initial_belief_H" if "initial_belief_H" in df.columns else "user_stance"
    if x_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{x_col}' column.")

    records = []
    for cond, grp in df.groupby("condition"):
        grp = grp.dropna(subset=[x_col, "H"])
        if len(grp) < 2:
            # not enough points to fit a line
            records.append({
                "condition": cond,
                "slope_chi": float("nan"),
                "intercept": float("nan"),
                "r_squared": float("nan"),
                "trials": len(grp),
            })
            continue

        x = grp[[x_col]].values
        y = grp["H"].values
        reg = LinearRegression().fit(x, y)
        records.append({
            "condition": cond,
            "slope_chi": float(reg.coef_[0]),
            "intercept": float(reg.intercept_),
            "r_squared": float(reg.score(x, y)),
            "trials": len(grp),
        })

    return pd.DataFrame(records)


def compute_response_surface(df: pd.DataFrame) -> pd.DataFrame:
    # average H per condition and user stance level
    x_col = "initial_belief_H" if "initial_belief_H" in df.columns else "user_stance"
    agg = (
        df.groupby(["condition", x_col])[["H"]]
        .mean()
        .reset_index()
    )
    return agg


def summarize_trajectory(df: pd.DataFrame) -> Dict[str, Any]:
    # summarize the user's final belief and the assistant's final expressed stance separately
    if df.empty:
        raise ValueError("Trajectory DataFrame is empty — nothing to summarize.")

    required = {"initial_belief_H", "H"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trajectory DataFrame missing columns: {missing}")

    initial_h = float(df["initial_belief_H"].iloc[0])
    final_user_h = float(df["user_H"].iloc[-1]) if "user_H" in df.columns else initial_h
    final_assistant_h = float(df["assistant_H"].iloc[-1]) if "assistant_H" in df.columns else float(df["H"].iloc[-1])
    initial_p_false, initial_p_truth = belief_probabilities_from_H(initial_h)
    final_user_p_false, final_user_p_truth = belief_probabilities_from_H(final_user_h)
    assistant_p_false, assistant_p_truth = belief_probabilities_from_H(final_assistant_h)
    return {
        "initial_belief_H": initial_h,
        "initial_belief_p_false": initial_p_false,
        "initial_belief_p_truth": initial_p_truth,
        "final_user_H": final_user_h,
        "final_user_p_false": final_user_p_false,
        "final_user_p_truth": final_user_p_truth,
        "final_assistant_H": final_assistant_h,
        "final_assistant_p_false": assistant_p_false,
        "final_assistant_p_truth": assistant_p_truth,
        # Backwards-compatible aliases: H now refers to the assistant's final expressed stance.
        "H": final_assistant_h,
        "p_false": assistant_p_false,
        "p_truth": assistant_p_truth,
    }
