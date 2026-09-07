"""HTTP exception handlers for consistent and auditable API errors.

The API returns the same error envelope for schema validation errors and
business validation errors. This gives clients a predictable contract and keeps
request IDs available for troubleshooting.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .schemas import ErrorResponse


def register_exception_handlers(app: FastAPI) -> None:
    """Attach API-wide exception handlers to a FastAPI application."""

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return _error_response(
            request,
            status_code=422,
            error_code="INVALID_INFERENCE_INPUT",
            message=str(exc),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            request,
            status_code=422,
            error_code="REQUEST_VALIDATION_ERROR",
            message="Request body does not match the API contract",
            details=_make_json_safe(exc.errors()),
        )


def _error_response(
    request: Request,
    *,
    status_code: int,
    error_code: str,
    message: str,
    details: list | None = None,
) -> JSONResponse:
    """Build a structured JSON error response.

    Parameters are explicit so each handler can choose a stable error code and
    status code without duplicating the response envelope.
    """
    payload = ErrorResponse(
        request_id=getattr(request.state, "request_id", ""),
        error_code=error_code,
        message=message,
        details=details or [],
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _make_json_safe(value):
    """Convert Pydantic error payloads into JSON-serializable values."""
    if isinstance(value, dict):
        return {key: _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, BaseException):
        return str(value)
    return value
