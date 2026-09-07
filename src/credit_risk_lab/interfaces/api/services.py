"""HTTP-facing services for prediction responses.

The service layer turns validated API payloads into application-layer scoring
calls and formats auditable HTTP responses. It deliberately avoids duplicating
model logic so notebooks, batch inference, and the API use the same scoring
path.
"""

from datetime import datetime, timezone
from time import perf_counter

import pandas as pd

from credit_risk_lab.application import InferenceInputValidator, RawLoanScorer
from credit_risk_lab.config.settings import settings

from .schemas import BatchPredictionResponse, LoanApplication, PredictionResponse


class PredictionApiService:
    """Coordinate API validation, application scoring, and response formatting."""

    def __init__(
        self,
        *,
        scorer: RawLoanScorer,
        validator: InferenceInputValidator,
    ):
        self.scorer = scorer
        self.validator = validator

    def predict_one(
        self,
        application: LoanApplication,
        *,
        request_id: str,
    ) -> PredictionResponse:
        """Validate one request, score it, and return an auditable response.

        Blocking validation errors are returned as HTTP 422 by the API
        exception handlers. Non-blocking warnings stay attached to the response
        so clients can monitor unusual but accepted values.
        """
        start = perf_counter()
        frame = pd.DataFrame([application.model_dump()])
        validation = self.validator.validate(frame)
        if validation.rejected_count:
            raise ValueError(validation.rejected_rows["rejection_reason"].iloc[0])

        result = self.scorer.score(validation.valid_frame)
        decision = int(result.decisions[0])
        probability = float(result.probabilities[0])
        return PredictionResponse(
            request_id=request_id,
            model_name=result.model_name,
            model_version=settings.project_version,
            probability_of_risk=probability,
            risk_decision=decision,
            risk_label="high_risk" if decision else "low_risk",
            risk_band=self._risk_band(probability, result.threshold),
            threshold=result.threshold,
            threshold_source="configs/settings.yaml:decision_threshold",
            validation_warnings=validation.warning_rows.get(
                "warning_reason",
                pd.Series(dtype=str),
            ).tolist(),
            scored_at_utc=datetime.now(timezone.utc),
            latency_ms=round((perf_counter() - start) * 1000, 3),
        )

    def predict_batch(
        self,
        applications: list[LoanApplication],
        *,
        request_id: str,
    ) -> BatchPredictionResponse:
        """Score a bounded list of applications while preserving input order."""
        start = perf_counter()
        predictions = [
            self.predict_one(application, request_id=f"{request_id}-{position:06d}")
            for position, application in enumerate(applications, start=1)
        ]

        return BatchPredictionResponse(
            request_id=request_id,
            model_name=predictions[0].model_name,
            model_version=settings.project_version,
            rows=len(predictions),
            predictions=predictions,
            scored_at_utc=datetime.now(timezone.utc),
            latency_ms=round((perf_counter() - start) * 1000, 3),
        )

    @staticmethod
    def _risk_band(probability: float, threshold: float) -> str:
        """Map a probability to an interpretation band for API consumers."""
        low_cutoff = min(0.10, threshold)
        if probability < low_cutoff:
            return "low_risk"
        if probability < threshold:
            return "watchlist"
        return "high_risk"
