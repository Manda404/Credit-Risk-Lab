"""Pydantic HTTP contracts for the inference API.

These schemas define the public API boundary. They validate raw loan
applications before they reach the application layer and document every
response field exposed to API clients through OpenAPI/Swagger.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LoanApplication(BaseModel):
    """Raw loan application accepted by prediction endpoints.

    The schema only contains fields that should be available at application
    time. The target column is intentionally excluded because production
    inference must never receive the true label.
    """

    model_config = ConfigDict(extra="forbid")

    person_age: float = Field(
        ge=18,
        le=100,
        description="Applicant age in years.",
    )
    person_gender: str = Field(
        min_length=1,
        description="Applicant gender as provided by the source system.",
    )
    person_education: str = Field(
        min_length=1,
        description="Highest education level declared by the applicant.",
    )
    person_income: float = Field(
        gt=0,
        description="Applicant annual income.",
    )
    person_emp_exp: float = Field(
        ge=0,
        le=80,
        description="Applicant employment experience in years.",
    )
    person_home_ownership: str = Field(
        min_length=1,
        description="Applicant housing ownership status.",
    )
    loan_amnt: float = Field(
        gt=0,
        description="Requested loan amount.",
    )
    loan_intent: str = Field(
        min_length=1,
        description="Declared purpose of the loan.",
    )
    loan_int_rate: float = Field(
        ge=0,
        le=100,
        description="Loan interest rate expressed as a percentage.",
    )
    loan_percent_income: float = Field(
        ge=0,
        description="Requested loan amount divided by applicant income.",
    )
    cb_person_cred_hist_length: float = Field(
        ge=0,
        description="Credit bureau history length in years.",
    )
    credit_score: float = Field(
        ge=300,
        le=900,
        description="Applicant credit score.",
    )
    previous_loan_defaults_on_file: str = Field(
        min_length=1,
        description="Whether previous loan defaults are present in the file.",
    )

    @model_validator(mode="after")
    def validate_experience(self) -> "LoanApplication":
        """Reject employment experience that is implausible for the age."""
        if self.person_emp_exp > self.person_age - 14:
            raise ValueError("Employment experience is implausible for age")
        return self


class BatchPredictionRequest(BaseModel):
    """Bounded batch prediction request.

    The API accepts small operational batches. Larger offline scoring jobs
    should use the batch inference workflow instead of this HTTP endpoint.
    """

    model_config = ConfigDict(extra="forbid")

    applications: list[LoanApplication] = Field(
        min_length=1,
        max_length=100,
        description="Loan applications to score, returned in the same order.",
    )


class PredictionResponse(BaseModel):
    """Auditable response returned for one scored application."""

    request_id: str = Field(description="Correlation ID propagated by the API.")
    model_name: str = Field(description="Name of the model used for scoring.")
    model_version: str = Field(description="Project or model package version.")
    probability_of_risk: float = Field(
        description="Predicted probability for the risky class."
    )
    risk_decision: int = Field(description="Binary decision after thresholding.")
    risk_label: str = Field(description="Human-readable decision label.")
    risk_band: str = Field(description="Three-level risk segment for interpretation.")
    threshold: float = Field(description="Decision threshold applied to probability.")
    threshold_source: str = Field(description="Configuration source of the threshold.")
    validation_warnings: list[str] = Field(
        default_factory=list,
        description="Non-blocking validation warnings detected before scoring.",
    )
    scored_at_utc: datetime = Field(description="UTC timestamp of the scoring event.")
    latency_ms: float = Field(description="End-to-end API scoring latency.")


class BatchPredictionResponse(BaseModel):
    """Batch response preserving input order and per-row auditability."""

    request_id: str = Field(description="Correlation ID for the batch request.")
    model_name: str = Field(description="Name of the model used for the batch.")
    model_version: str = Field(description="Project or model package version.")
    rows: int = Field(description="Number of scored applications.")
    predictions: list[PredictionResponse] = Field(
        description="Per-application predictions in input order."
    )
    scored_at_utc: datetime = Field(description="UTC timestamp of the batch response.")
    latency_ms: float = Field(description="End-to-end batch scoring latency.")


class HealthResponse(BaseModel):
    """Lightweight liveness response for orchestration systems."""

    status: str = Field(description="Liveness status of the API process.")
    service: str = Field(description="Service name from project settings.")
    environment: str = Field(description="Runtime environment name.")
    version: str = Field(description="Project version exposed by the API.")


class ReadyResponse(BaseModel):
    """Readiness response with model artifact metadata."""

    status: str = Field(description="Readiness status after model loading.")
    environment: str = Field(description="Runtime environment name.")
    model: str = Field(description="Loaded model name from bundle metadata.")
    model_version: str = Field(description="Project or model package version.")
    model_bundle_path: str = Field(description="Filesystem path of the model bundle.")
    model_bundle_sha256: str = Field(
        description="SHA256 digest of the loaded model bundle."
    )
    threshold: float = Field(description="Decision threshold used by the loaded model.")


class ErrorResponse(BaseModel):
    """Consistent API error envelope returned by all HTTP error handlers."""

    request_id: str = Field(
        description="Correlation ID attached to the failed request."
    )
    error_code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable error summary.")
    details: list = Field(
        default_factory=list,
        description="Optional structured validation details.",
    )
