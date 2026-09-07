"""Small FastAPI entry point: lifecycle and HTTP routes only."""

from contextlib import asynccontextmanager
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request

from credit_risk_lab.config.settings import settings
from .api_models import LoanApplication, PredictionResponse
from .api_service import get_scorer, predict_application


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model at startup and release the process cache at shutdown."""
    get_scorer()
    yield
    get_scorer.cache_clear()


app = FastAPI(
    title="Credit Risk Lab Inference API",
    version=settings.project_version,
    description="Educational risk scoring API. It must not make real lending decisions.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return service liveness for Docker and cloud health checks."""
    return {
        "status": "ok",
        "service": settings.project_name,
        "environment": settings.environment,
    }


@app.get("/ready")
def ready() -> dict[str, str]:
    """Confirm that the model bundle is loaded."""
    scorer = get_scorer()
    return {
        "status": "ready",
        "environment": settings.environment,
        "model": scorer.model_scorer.bundle["metadata"]["model_name"],
    }


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(application: LoanApplication, request: Request) -> PredictionResponse:
    """Score one validated application."""
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    try:
        return predict_application(application, request_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
