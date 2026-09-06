"""HTTP request and response schemas for the inference API."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, model_validator


class LoanApplication(BaseModel):
    """Raw application fields accepted by ``POST /v1/predict``."""

    model_config = ConfigDict(extra="forbid")

    person_age: float = Field(ge=18, le=100)
    person_gender: str = Field(min_length=1)
    person_education: str = Field(min_length=1)
    person_income: float = Field(gt=0)
    person_emp_exp: float = Field(ge=0, le=80)
    person_home_ownership: str = Field(min_length=1)
    loan_amnt: float = Field(gt=0)
    loan_intent: str = Field(min_length=1)
    loan_int_rate: float = Field(ge=0, le=100)
    loan_percent_income: float = Field(ge=0)
    cb_person_cred_hist_length: float = Field(ge=0)
    credit_score: float = Field(ge=300, le=900)
    previous_loan_defaults_on_file: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_experience(self) -> "LoanApplication":
        """Reject employment experience that is implausible for the age."""
        if self.person_emp_exp > self.person_age - 14:
            raise ValueError("Employment experience is implausible for age")
        return self


class PredictionResponse(BaseModel):
    """Auditable response returned for one application."""

    request_id: str
    model_name: str
    model_version: str
    probability_of_risk: float
    risk_decision: int
    risk_label: str
    threshold: float
    threshold_source: str
    scored_at_utc: datetime
    latency_ms: float
