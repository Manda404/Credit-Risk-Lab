"""Decision-threshold diagnostics for credit-risk scoring."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class ThresholdAnalysisConfig:
    """Configuration for threshold grid diagnostics."""

    start: float = 0.02
    stop: float = 0.42
    step: float = 0.04
    false_alert_cost: float = 1.0
    missed_high_risk_cost: float = 5.0


class ThresholdAnalyzer:
    """Analyze the operational trade-off produced by score thresholds.

    In this project, class 1 means high-risk/default-like behavior. A lower
    threshold catches more risky loans but also creates more false alerts.
    """

    def __init__(self, config: ThresholdAnalysisConfig | None = None):
        self.config = config or ThresholdAnalysisConfig()

    def grid(self, y_true, probabilities: np.ndarray) -> pd.DataFrame:
        """Return credit-risk threshold diagnostics across a threshold grid."""
        target = np.asarray(y_true, dtype=int)
        probabilities = np.asarray(probabilities, dtype=float)
        if len(target) != len(probabilities) or len(target) == 0:
            raise ValueError(
                "Targets and probabilities must have the same non-zero length"
            )
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Probabilities must be between 0 and 1")

        rows = []
        thresholds = np.round(
            np.arange(self.config.start, self.config.stop, self.config.step),
            3,
        )
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                target,
                predictions,
                labels=[0, 1],
            ).ravel()
            alerts = tp + fp
            actual_high_risk = tp + fn
            rows.append(
                {
                    "threshold": float(threshold),
                    "high_risk_caught": int(tp),
                    "high_risk_missed": int(fn),
                    "false_alerts": int(fp),
                    "true_low_risk": int(tn),
                    "precision": round(tp / alerts, 3) if alerts else 0.0,
                    "recall": (
                        round(tp / actual_high_risk, 3) if actual_high_risk else 0.0
                    ),
                    "alert_rate": round(alerts / len(target), 3),
                    "approval_rate": round((tn + fn) / len(target), 3),
                    "estimated_cost": float(
                        fp * self.config.false_alert_cost
                        + fn * self.config.missed_high_risk_cost
                    ),
                }
            )
        return pd.DataFrame(rows)
