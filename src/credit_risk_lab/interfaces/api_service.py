"""Model loading and prediction logic used by the HTTP routes."""

from datetime import datetime, timezone
from functools import lru_cache
from time import perf_counter
import pandas as pd

from credit_risk_lab.application import RawLoanScorer
from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.modeling import load_model_bundle
from .api_models import LoanApplication, PredictionResponse


@lru_cache(maxsize=1)
def get_scorer() -> RawLoanScorer:
    """Load the trusted model once per API process."""
    return RawLoanScorer(
        load_model_bundle(settings.model_bundle_path),
        threshold=settings.decision_threshold,
    )


def predict_application(application: LoanApplication, request_id: str) -> PredictionResponse:
    """Engineer features, run inference, and build an auditable response."""
    start = perf_counter()
    result = get_scorer().score(pd.DataFrame([application.model_dump()]))
    decision = int(result.decisions[0])
    return PredictionResponse(
        request_id=request_id,
        model_name=result.model_name,
        model_version=settings.project_version,
        probability_of_risk=float(result.probabilities[0]),
        risk_decision=decision,
        risk_label="high_risk" if decision else "low_risk",
        threshold=result.threshold,
        threshold_source="configs/settings.yaml:decision_threshold",
        scored_at_utc=datetime.now(timezone.utc),
        latency_ms=round((perf_counter() - start) * 1000, 3),
    )
