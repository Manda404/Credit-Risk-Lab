"""Backward-compatible API service imports for older project code.

New code should use ``credit_risk_lab.interfaces.api.services`` and
``credit_risk_lab.interfaces.api.dependencies`` directly. This module preserves
the old ``predict_application`` entry point to avoid breaking notebooks,
scripts, or tests that still import it.
"""

from credit_risk_lab.interfaces.api.dependencies import get_input_validator, get_scorer
from credit_risk_lab.interfaces.api.services import PredictionApiService


def predict_application(application, request_id: str):
    """Score one application through the current API service implementation."""
    return PredictionApiService(
        scorer=get_scorer(),
        validator=get_input_validator(),
    ).predict_one(application, request_id=request_id)


__all__ = [
    "PredictionApiService",
    "get_input_validator",
    "get_scorer",
    "predict_application",
]
