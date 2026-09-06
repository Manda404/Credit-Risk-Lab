"""Loan application domain objects and schema constants."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoanApplicationRecord:
    """Raw application fields known at decision time."""

    person_age: float
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: float
    person_home_ownership: str
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    credit_score: float
    previous_loan_defaults_on_file: str

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "LoanApplicationRecord":
        """Build a domain record from an API, CSV, or notebook row."""
        return cls(**{field: values[field] for field in LoanSchema.raw_feature_columns})


@dataclass(frozen=True)
class LoanSchema:
    """Column names used by the loan-approval use cases."""

    target_column: str = "loan_status"

    raw_feature_columns = (
        "person_age",
        "person_gender",
        "person_education",
        "person_income",
        "person_emp_exp",
        "person_home_ownership",
        "loan_amnt",
        "loan_intent",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "credit_score",
        "previous_loan_defaults_on_file",
    )

    required_training_columns = raw_feature_columns + (target_column,)
