"""HTTP routes for credit-risk prediction use cases.

Routes should only translate HTTP inputs into service calls. Validation rules,
feature engineering, and model inference remain outside this router so the API
stays aligned with the clean architecture of the project.
"""

from fastapi import APIRouter, Depends, Request

from credit_risk_lab.application import InferenceInputValidator, RawLoanScorer

from ..dependencies import get_input_validator, get_scorer
from ..schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    LoanApplication,
    PredictionResponse,
)
from ..services import PredictionApiService

router = APIRouter(
    prefix="/v1",
    tags=["predictions"],
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Invalid request schema or rejected inference input.",
        }
    },
)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    name="score_one_application",
    summary="Score one loan application",
    description=(
        "Validate one raw loan application, apply the trained feature pipeline, "
        "and return an auditable credit-risk prediction."
    ),
)
def predict(
    application: LoanApplication,
    request: Request,
    scorer: RawLoanScorer = Depends(get_scorer),
    validator: InferenceInputValidator = Depends(get_input_validator),
) -> PredictionResponse:
    """Score one loan application through the public HTTP contract."""
    service = PredictionApiService(scorer=scorer, validator=validator)
    return service.predict_one(
        application,
        request_id=request.state.request_id,
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    name="score_batch_applications",
    summary="Score a bounded batch of loan applications",
    description=(
        "Score multiple raw loan applications while preserving input order and "
        "returning one auditable prediction per row."
    ),
)
def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
    scorer: RawLoanScorer = Depends(get_scorer),
    validator: InferenceInputValidator = Depends(get_input_validator),
) -> BatchPredictionResponse:
    """Score a bounded batch of loan applications through the HTTP contract."""
    service = PredictionApiService(scorer=scorer, validator=validator)
    return service.predict_batch(
        payload.applications,
        request_id=request.state.request_id,
    )
