"""Backward-compatible schema imports for older project code.

The canonical HTTP schemas now live in ``credit_risk_lab.interfaces.api``.
This module keeps previous imports working while the project transitions to the
package-based API structure.
"""

from credit_risk_lab.interfaces.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    LoanApplication,
    PredictionResponse,
    ReadyResponse,
)

__all__ = [
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "ErrorResponse",
    "HealthResponse",
    "LoanApplication",
    "PredictionResponse",
    "ReadyResponse",
]
