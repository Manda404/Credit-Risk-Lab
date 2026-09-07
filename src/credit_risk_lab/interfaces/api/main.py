"""FastAPI application factory for the HTTP delivery layer.

This module wires routers, middleware, exception handlers, and startup logic.
It does not perform model scoring directly; prediction work is delegated to
the API service, which then calls the application layer.
"""

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request

from credit_risk_lab.config.settings import settings

from .dependencies import get_input_validator, get_scorer
from .exception_handlers import register_exception_handlers
from .routers import health, predictions


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm process-level runtime dependencies and clear caches on shutdown.

    Loading the model bundle at startup makes readiness failures visible before
    the first prediction request reaches the API.
    """
    get_scorer()
    get_input_validator()
    yield
    get_scorer.cache_clear()
    get_input_validator.cache_clear()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    The function is kept separate from the module-level ``app`` object to make
    tests and future deployment factories easier to customize.
    """
    api = FastAPI(
        title="Credit Risk Lab Inference API",
        version=settings.project_version,
        description=(
            "Educational credit-risk scoring API. It must not make real lending "
            "decisions without model-risk governance."
        ),
        lifespan=lifespan,
    )

    @api.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        """Attach a correlation ID to each request and response."""
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    register_exception_handlers(api)
    api.include_router(health.router)
    api.include_router(predictions.router)
    return api


app = create_app()
