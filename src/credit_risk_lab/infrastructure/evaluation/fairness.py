"""Group diagnostics for model governance, not legal fairness certification."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, recall_score


def fairness_report(
    y_true,
    probabilities: np.ndarray,
    sensitive: pd.DataFrame,
    threshold: float,
    *,
    min_group_size: int = 30,
) -> pd.DataFrame:
    """Compute selection, error, and calibration diagnostics per group.

    Small groups are retained but flagged because point estimates without
    adequate support are unstable. Results support human/legal review and must
    not be interpreted as proof that a lending policy is non-discriminatory.
    """
    target = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(target) != len(probabilities) or len(target) != len(sensitive):
        raise ValueError("Target, probabilities, and sensitive rows must align")
    decisions = (probabilities >= threshold).astype(int)
    rows = []
    for attribute in sensitive.columns:
        values = sensitive[attribute].astype("string").fillna("<missing>")
        for group in values.unique():
            mask = values.eq(group).to_numpy()
            group_y, group_d, group_p = (
                target[mask],
                decisions[mask],
                probabilities[mask],
            )
            negatives = group_y == 0
            rows.append(
                {
                    "attribute": attribute,
                    "group": str(group),
                    "rows": int(mask.sum()),
                    "small_group": bool(mask.sum() < min_group_size),
                    "positive_rate": float(group_y.mean()),
                    "selection_rate": float(group_d.mean()),
                    "true_positive_rate": float(
                        recall_score(group_y, group_d, zero_division=0)
                    ),
                    "false_positive_rate": (
                        float(group_d[negatives].mean()) if negatives.any() else np.nan
                    ),
                    "mean_probability": float(group_p.mean()),
                    "observed_rate": float(group_y.mean()),
                    "calibration_gap": float(abs(group_p.mean() - group_y.mean())),
                    "brier": float(brier_score_loss(group_y, group_p)),
                }
            )
    return pd.DataFrame(rows)
