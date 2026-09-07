"""Explicit model evaluation services."""

import numpy as np
import pandas as pd

from .classification import (
    calibration_table,
    classification_metrics,
    decile_table,
    find_optimal_threshold,
)
from .fairness import fairness_report


class CreditRiskModelEvaluator:
    """Evaluate probabilities, thresholds, and ranking diagnostics."""

    def optimal_threshold(self, y_true, probabilities: np.ndarray) -> float:
        """Return the threshold selected on the provided validation target."""
        return find_optimal_threshold(y_true, probabilities)

    def metrics(
        self, y_true, probabilities: np.ndarray, threshold: float
    ) -> dict[str, float]:
        """Return classification metrics for one operating threshold."""
        return classification_metrics(y_true, probabilities, threshold)

    def metrics_frame(
        self, model_name: str, y_true, probabilities: np.ndarray, threshold: float
    ) -> pd.DataFrame:
        """Return metrics as a single-row DataFrame."""
        return pd.DataFrame(
            [
                {
                    "model": model_name,
                    "threshold": threshold,
                    **self.metrics(y_true, probabilities, threshold),
                }
            ]
        )

    def deciles(
        self, y_true, probabilities: np.ndarray, bins: int = 10
    ) -> pd.DataFrame:
        """Return a lift and cumulative-gain table."""
        return decile_table(y_true, probabilities, bins=bins)


class CalibrationEvaluator:
    """Evaluate probability calibration."""

    def __init__(self, bins: int = 10):
        self.bins = bins

    def evaluate(self, y_true, probabilities: np.ndarray) -> pd.DataFrame:
        """Return a calibration table."""
        return calibration_table(y_true, probabilities, bins=self.bins)


class FairnessEvaluator:
    """Evaluate group diagnostics for sensitive attributes."""

    def __init__(self, min_group_size: int = 30):
        self.min_group_size = min_group_size

    def evaluate(
        self,
        y_true,
        probabilities: np.ndarray,
        sensitive: pd.DataFrame,
        threshold: float,
    ) -> pd.DataFrame:
        """Return group-level diagnostic metrics."""
        return fairness_report(
            y_true,
            probabilities,
            sensitive,
            threshold,
            min_group_size=self.min_group_size,
        )
