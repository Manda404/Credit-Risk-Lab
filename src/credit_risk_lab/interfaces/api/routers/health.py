"""Health and readiness routes for deployment environments.

The liveness endpoint checks the API process. The readiness endpoint checks
that the configured model bundle can be loaded and identified, which is the
minimum condition required before serving predictions.
"""

from fastapi import APIRouter

from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.modeling import sha256_file

from ..dependencies import get_scorer
from ..schemas import ErrorResponse, HealthResponse, ReadyResponse

router = APIRouter(
    tags=["health"],
    responses={
        500: {
            "model": ErrorResponse,
            "description": "Unexpected API or model artifact error.",
        }
    },
)


@router.get(
    "/health",
    response_model=HealthResponse,
    name="health_check",
    summary="Check API liveness",
    description="Return a lightweight liveness response for containers and monitoring.",
)
def health() -> HealthResponse:
    """Return process liveness metadata for containers and monitors."""
    return HealthResponse(
        status="ok",
        service=settings.project_name,
        environment=settings.environment,
        version=settings.project_version,
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    name="readiness_check",
    summary="Check model readiness",
    description=(
        "Confirm that the model bundle can be loaded and expose auditable model "
        "artifact metadata."
    ),
)
def ready() -> ReadyResponse:
    """Return readiness metadata for the loaded model artifact."""
    scorer = get_scorer()
    metadata = scorer.model_scorer.bundle["metadata"]
    return ReadyResponse(
        status="ready",
        environment=settings.environment,
        model=str(metadata["model_name"]),
        model_version=settings.project_version,
        model_bundle_path=str(settings.model_bundle_path),
        model_bundle_sha256=sha256_file(settings.model_bundle_path),
        threshold=scorer.model_scorer.threshold,
    )
