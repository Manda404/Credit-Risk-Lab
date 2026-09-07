"""Validated inference service for trusted local model bundles."""

from dataclasses import dataclass
from pathlib import Path
import random
import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.feature_engineering import LoanFeatureEngineer
from credit_risk_lab.shared.logging import setup_logger


@dataclass(frozen=True)
class ScoringResult:
    """Probabilities and binary risk decisions returned in input row order."""

    probabilities: np.ndarray
    decisions: np.ndarray
    model_name: str
    threshold: float


@dataclass(frozen=True)
class BatchInferenceResult:
    """Submission output and destination path produced by batch inference."""

    submission: pd.DataFrame
    output_path: Path


@dataclass(frozen=True)
class RealtimeInferenceEvent:
    """One simulated online prediction event."""

    request_id: str
    row_position: int
    probability_of_risk: float
    risk_decision: int
    threshold: float
    model_name: str
    pause_seconds: float


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
        self.feature_engineer = feature_engineer or LoanFeatureEngineer(
            logger_name="api_features"
        )

    def score(self, raw_frame: pd.DataFrame) -> ScoringResult:
        """Transform raw loan applications and return risk predictions.

        The target, when accidentally present, is removed before processing. All
        learned preprocessing remains inside the persisted preprocessor and is
        never fitted by the API.
        """
        raw_frame = raw_frame.drop(columns=["loan_status"], errors="ignore")
        engineered = self.feature_engineer.transform(raw_frame, is_training=False)
        return self.model_scorer.score(engineered)


class BatchInferenceRunner:
    """Score a raw dataset in batch and persist a compact submission file."""

    CONTEXT_COLUMNS = (
        "person_age",
        "person_income",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "loan_intent",
        "cb_person_default_on_file",
    )

    def __init__(
        self,
        scorer: RawLoanScorer,
        *,
        target_column: str = settings.target_column,
        logger_name: str = "batch_inference",
    ):
        self.scorer = scorer
        self.target_column = target_column
        self.logger = setup_logger(logger_name)

    def predict(
        self,
        raw_frame: pd.DataFrame,
        *,
        output_path: Path,
        limit: int | None = None,
    ) -> BatchInferenceResult:
        """Score the first `limit` rows, save `submission.csv`, and return it."""
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive when provided")

        frame = raw_frame.head(limit).copy() if limit is not None else raw_frame.copy()
        self.logger.info("Starting batch inference on {} rows", len(frame))

        scoring = self.scorer.score(frame)
        submission = self._build_submission(frame, scoring)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        self.logger.info("Batch inference file saved to {}", output_path)

        return BatchInferenceResult(submission=submission, output_path=output_path)

    def _build_submission(
        self,
        frame: pd.DataFrame,
        scoring: ScoringResult,
    ) -> pd.DataFrame:
        submission = pd.DataFrame(
            {
                "request_id": [f"batch-{i:06d}" for i in range(len(frame))],
                "source_row_index": frame.index.to_numpy(),
                "model_name": scoring.model_name,
                "threshold": scoring.threshold,
                "probability_of_risk": np.round(scoring.probabilities, 6),
                "risk_decision": scoring.decisions,
                "risk_label": np.where(
                    scoring.decisions == 1,
                    "high_risk",
                    "low_risk",
                ),
                "risk_band": self._risk_band(scoring.probabilities),
            }
        )
        available_context = [c for c in self.CONTEXT_COLUMNS if c in frame.columns]
        if available_context:
            submission = pd.concat(
                [submission, frame[available_context].reset_index(drop=True)],
                axis=1,
            )
        if self.target_column in frame.columns:
            actual = frame[self.target_column].reset_index(drop=True)
            submission["actual_label"] = actual
            submission["is_correct"] = submission["risk_decision"].eq(actual)
        return submission

    @staticmethod
    def _risk_band(probabilities: np.ndarray) -> np.ndarray:
        return np.select(
            [
                probabilities < 0.20,
                probabilities < 0.50,
                probabilities < 0.80,
            ],
            ["low", "medium", "high"],
            default="critical",
        )


class RealtimeInferenceSimulator:
    """Simulate online requests without starting an API server."""

    def __init__(
        self,
        scorer: RawLoanScorer,
        *,
        min_pause_seconds: float = 1.0,
        max_pause_seconds: float = 5.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        logger_name: str = "realtime_inference",
    ):
        if min_pause_seconds < 0 or max_pause_seconds < min_pause_seconds:
            raise ValueError("Invalid pause interval")
        self.scorer = scorer
        self.min_pause_seconds = min_pause_seconds
        self.max_pause_seconds = max_pause_seconds
        self.sleep_fn = sleep_fn
        self.logger = setup_logger(logger_name)

    def stream(
        self,
        raw_frame: pd.DataFrame,
        *,
        limit: int = 5,
        request_prefix: str = "rt",
        print_events: bool = True,
    ) -> pd.DataFrame:
        """Score rows one by one with random pauses between requests."""
        if limit <= 0:
            raise ValueError("limit must be positive")

        events = []
        rows = raw_frame.head(limit)
        self.logger.info(
            "Starting realtime inference simulation on {} requests", len(rows)
        )

        for position, (_, row) in enumerate(rows.iterrows(), start=1):
            pause = random.uniform(self.min_pause_seconds, self.max_pause_seconds)
            request_id = f"{request_prefix}-{position:06d}"

            self.logger.info("Receiving request {}", request_id)
            scoring = self.scorer.score(pd.DataFrame([row]))

            event = RealtimeInferenceEvent(
                request_id=request_id,
                row_position=position,
                probability_of_risk=float(scoring.probabilities[0]),
                risk_decision=int(scoring.decisions[0]),
                threshold=float(scoring.threshold),
                model_name=scoring.model_name,
                pause_seconds=round(pause, 3),
            )
            events.append(event)

            message = (
                f"{event.request_id} | probability={event.probability_of_risk:.4f} "
                f"| decision={event.risk_decision} | threshold={event.threshold:.3f} "
                f"| next_pause={event.pause_seconds:.2f}s"
            )
            self.logger.info(message)
            if print_events:
                print(message)

            if position < len(rows):
                self.sleep_fn(pause)

        self.logger.info("Realtime inference simulation completed")
        return pd.DataFrame([event.__dict__ for event in events])
