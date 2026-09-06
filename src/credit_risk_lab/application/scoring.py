"""Validated inference service for trusted local model bundles."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from credit_risk_lab.infrastructure.feature_engineering import LoanFeatureEngineer


@dataclass(frozen=True)
class ScoringResult:
    """Probabilities and binary risk decisions returned in input row order."""

    probabilities: np.ndarray
    decisions: np.ndarray
    model_name: str
    threshold: float


class ModelScorer:
    """Enforce the training schema before preprocessing and prediction."""

    def __init__(self, bundle: dict, *, threshold: float | None = None):
        self.bundle = bundle
        self.threshold = float(bundle["threshold"] if threshold is None else threshold)
        if not 0 <= self.threshold <= 1:
            raise ValueError("Serving threshold must be between 0 and 1")
        self.schema = list(bundle["metadata"].get("feature_schema", []))
        if not self.schema:
            raise ValueError("Artifact has no feature schema")

    def score(self, frame: pd.DataFrame) -> ScoringResult:
        """Validate columns, preserve order, and apply the persisted threshold."""
        if frame.empty:
            raise ValueError("Cannot score an empty dataset")
        missing = sorted(set(self.schema).difference(frame.columns))
        if missing:
            raise ValueError(f"Missing inference features: {missing}")
        features = frame.loc[:, self.schema]
        transformed = self.bundle["preprocessor"].transform(features)
        probabilities = self.bundle["model"].predict_proba(transformed)
        threshold = self.threshold
        return ScoringResult(
            probabilities=np.asarray(probabilities),
            decisions=(np.asarray(probabilities) >= threshold).astype(int),
            model_name=str(self.bundle["metadata"].get("model_name", "unknown")),
            threshold=threshold,
        )


class RawLoanScorer:
    """Run deterministic feature engineering before the persisted ML pipeline."""

    def __init__(
        self,
        bundle: dict,
        feature_engineer: LoanFeatureEngineer | None = None,
        *,
        threshold: float | None = None,
    ):
        self.model_scorer = ModelScorer(bundle, threshold=threshold)
        self.feature_engineer = feature_engineer or LoanFeatureEngineer(logger_name="api_features")

    def score(self, raw_frame: pd.DataFrame) -> ScoringResult:
        """Transform raw loan applications and return risk predictions.

        The target, when accidentally present, is removed before processing. All
        learned preprocessing remains inside the persisted preprocessor and is
        never fitted by the API.
        """
        raw_frame = raw_frame.drop(columns=["loan_status"], errors="ignore")
        engineered = self.feature_engineer.transform(raw_frame, is_training=False)
        return self.model_scorer.score(engineered)
