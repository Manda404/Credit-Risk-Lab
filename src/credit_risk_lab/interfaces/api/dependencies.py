"""FastAPI dependency providers for inference endpoints.

Dependency functions are the bridge between the HTTP layer and the existing
application services. They load shared runtime objects once per process so the
routes can stay focused on request/response orchestration.
"""

from functools import lru_cache

from credit_risk_lab.application import InferenceInputValidator, RawLoanScorer
from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.modeling import JoblibModelBundleRepository


@lru_cache(maxsize=1)
def get_scorer() -> RawLoanScorer:
    """Return a cached scorer configured from project settings.

    The scorer wraps the persisted model bundle and applies the same decision
    threshold used by notebooks, batch inference, and the MLOps pipeline.
    Caching avoids reloading the model artifact for every HTTP request.
    """
    repository = JoblibModelBundleRepository()
    return RawLoanScorer(
        repository.load(settings.model_bundle_path),
        threshold=settings.decision_threshold,
    )


@lru_cache(maxsize=1)
def get_input_validator() -> InferenceInputValidator:
    """Return the cached application-level input validator.

    The validator is intentionally reused from the application layer so API,
    batch inference, and realtime simulation reject inputs with the same rules.
    """
    return InferenceInputValidator()
