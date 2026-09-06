"""External holdout evaluation workflow."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from credit_risk_lab.application import RawLoanScorer
from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.evaluation import (
    calibration_table,
    classification_metrics,
    fairness_report,
)
from credit_risk_lab.infrastructure.modeling import load_model_bundle

from ._shared import load_external_test_dataset


@dataclass(frozen=True)
class ExternalEvaluationResult:
    external_test: pd.DataFrame
    probabilities: np.ndarray
    metrics: pd.DataFrame
    calibration: pd.DataFrame
    fairness: pd.DataFrame


def run_external_evaluation_workflow() -> ExternalEvaluationResult:
    """Evaluate the persisted winner on the untouched raw external holdout."""
    external_test = load_external_test_dataset()
    scorer = RawLoanScorer(load_model_bundle(settings.model_bundle_path))
    result = scorer.score(external_test)
    probabilities = np.asarray(result.probabilities)
    y_true = external_test[settings.target_column].astype(int)
    metrics = pd.DataFrame(
        [
            {
                "model": result.model_name,
                "threshold": result.threshold,
                **classification_metrics(y_true, probabilities, result.threshold),
            }
        ]
    )
    metrics.to_csv(settings.reports_dir / "external_test_metrics.csv", index=False)
    calibration = calibration_table(y_true, probabilities)
    sensitive = external_test[
        [c for c in settings.sensitive_columns if c in external_test]
    ]
    fairness = fairness_report(y_true, probabilities, sensitive, result.threshold)
    return ExternalEvaluationResult(
        external_test, probabilities, metrics, calibration, fairness
    )
