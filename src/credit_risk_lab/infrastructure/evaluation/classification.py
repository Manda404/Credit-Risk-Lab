"""Credit-risk classification metrics and threshold selection."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def find_optimal_threshold(y_true, probabilities: np.ndarray) -> float:
    """Choose the validation threshold maximizing Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])


def find_cost_sensitive_threshold(
    y_true,
    probabilities: np.ndarray,
    *,
    false_positive_cost: float,
    false_negative_cost: float,
) -> float:
    """Choose the validation threshold minimizing an explicit error cost.

    Cost orientation depends on the documented positive class. For this lab,
    positive means risk/default; accepting a predicted negative that defaults is
    a false negative. Costs must be approved by business and model-risk owners.
    """
    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError("Error costs must be non-negative")
    target = np.asarray(y_true, dtype=int)
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    costs = []
    for threshold in candidates:
        prediction = np.asarray(probabilities) >= threshold
        false_positives = np.sum((prediction == 1) & (target == 0))
        false_negatives = np.sum((prediction == 0) & (target == 1))
        costs.append(
            false_positives * false_positive_cost
            + false_negatives * false_negative_cost
        )
    return float(candidates[int(np.argmin(costs))])


def calibration_table(
    y_true, probabilities: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    """Aggregate confidence and observed frequency into equal-width probability bins.

    Empty bins are omitted from the returned table and from the weighted ECE.
    Equal-width bins keep the meaning of probability ranges explicit, which is
    preferable for a calibration diagnostic.
    """
    if bins < 2:
        raise ValueError("bins must be at least 2")
    target = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(target) != len(probabilities) or len(target) == 0:
        raise ValueError("Targets and probabilities must have the same non-zero length")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be between 0 and 1")

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.clip(np.digitize(probabilities, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        confidence = float(probabilities[mask].mean())
        observed_rate = float(target[mask].mean())
        rows.append(
            {
                "bin": bin_id + 1,
                "lower_bound": float(edges[bin_id]),
                "upper_bound": float(edges[bin_id + 1]),
                "rows": int(mask.sum()),
                "mean_probability": confidence,
                "observed_rate": observed_rate,
                "absolute_gap": abs(observed_rate - confidence),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true, probabilities: np.ndarray, bins: int = 10
) -> float:
    """Compute weighted Expected Calibration Error (ECE).

    ECE is a useful summary but depends on the binning choice. It must always be
    interpreted alongside a calibration curve and the Brier score.
    """
    table = calibration_table(y_true, probabilities, bins=bins)
    return float(np.average(table["absolute_gap"], weights=table["rows"]))


def classification_metrics(
    y_true, probabilities: np.ndarray, threshold: float, calibration_bins: int = 10
) -> dict[str, float]:
    """Return the ten complementary professional classification metrics.

    Threshold-dependent metrics describe decisions at one operating point.
    ROC-AUC describes ranking. Log loss and Brier evaluate probabilities, while
    ECE summarizes calibration. No single metric is sufficient on its own.
    """
    predictions = (probabilities >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "cohen_kappa": float(cohen_kappa_score(y_true, predictions)),
        "ece": expected_calibration_error(y_true, probabilities, bins=calibration_bins),
        # Credit-risk complements retained for ranking and imbalance analysis.
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "ks": float(np.max(tpr - fpr)),
        "gini": float(2 * roc_auc_score(y_true, probabilities) - 1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
    }


def decile_table(y_true, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Build a standard credit-risk lift and cumulative-gain table."""
    frame = pd.DataFrame({"target": np.asarray(y_true), "probability": probabilities})
    frame = frame.sort_values("probability", ascending=False).reset_index(drop=True)
    frame["decile"] = pd.qcut(frame.index, q=bins, labels=range(1, bins + 1))
    grouped = frame.groupby("decile", observed=True).agg(
        rows=("target", "size"),
        positives=("target", "sum"),
        average_score=("probability", "mean"),
    )
    grouped["positive_rate"] = grouped["positives"] / grouped["rows"]
    grouped["lift"] = grouped["positive_rate"] / frame["target"].mean()
    grouped["cumulative_gain"] = (
        grouped["positives"].cumsum() / grouped["positives"].sum()
    )
    return grouped.reset_index()


def lift_gain_table(y_true, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Return decile-level lift, gain, and accumulation metrics."""
    table = decile_table(y_true, probabilities, bins=bins).copy()
    total_rows = table["rows"].sum()
    total_positives = table["positives"].sum()
    table["cumulative_rows"] = table["rows"].cumsum()
    table["sample_fraction"] = table["cumulative_rows"] / total_rows
    table["cumulative_positives"] = table["positives"].cumsum()
    table["cumulative_capture_rate"] = table["cumulative_positives"] / total_positives
    table["cumulative_lift"] = (
        table["cumulative_capture_rate"] / table["sample_fraction"]
    )
    return table


def bootstrap_metric_intervals(
    y_true,
    probabilities: np.ndarray,
    threshold: float,
    *,
    n_bootstrap: int = 300,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate confidence intervals for key classification metrics."""
    if n_bootstrap < 10:
        raise ValueError("n_bootstrap must be at least 10")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    target = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(random_state)
    metrics_by_name: dict[str, list[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "ks": [],
        "f1": [],
        "mcc": [],
        "balanced_accuracy": [],
    }
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(target), size=len(target))
        sampled_target = target[index]
        if len(np.unique(sampled_target)) < 2:
            continue
        sampled_probabilities = probabilities[index]
        metrics = classification_metrics(
            sampled_target, sampled_probabilities, threshold
        )
        for metric in metrics_by_name:
            metrics_by_name[metric].append(metrics[metric])
    alpha = 1 - confidence_level
    rows = []
    point_estimates = classification_metrics(target, probabilities, threshold)
    for metric, values in metrics_by_name.items():
        if not values:
            continue
        rows.append(
            {
                "metric": metric,
                "estimate": point_estimates[metric],
                "lower": float(np.quantile(values, alpha / 2)),
                "upper": float(np.quantile(values, 1 - alpha / 2)),
                "confidence_level": confidence_level,
                "bootstrap_samples": len(values),
            }
        )
    return pd.DataFrame(rows)
