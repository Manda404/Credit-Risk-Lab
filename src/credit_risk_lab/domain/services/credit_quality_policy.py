"""Business-plausibility policies for the loan dataset."""

from collections.abc import Iterable
from typing import Protocol

from credit_risk_lab.domain.entities import LoanSchema


REQUIRED_COLUMNS = set(LoanSchema.required_training_columns)


class ColumnarFrame(Protocol):
    """Minimal frame protocol used by the domain policy."""

    columns: Iterable[str]


def missing_required_columns(frame: ColumnarFrame) -> list[str]:
    """Return missing required training columns in stable order."""
    return sorted(REQUIRED_COLUMNS.difference(frame.columns))


def validate_training_schema(frame) -> None:
    """Raise when required columns or the binary target contract are invalid."""
    missing = missing_required_columns(frame)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    target_values = set(frame[LoanSchema.target_column].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"{LoanSchema.target_column} must be binary, received {target_values}"
        )


def invalid_age_mask(frame):
    """Rows outside the accepted application age range."""
    return ~frame["person_age"].between(18, 100)


def invalid_experience_mask(frame):
    """Rows with impossible employment experience for the stated age."""
    return (
        (frame["person_emp_exp"] < 0)
        | (frame["person_emp_exp"] > 80)
        | (frame["person_emp_exp"] > frame["person_age"] - 14)
    )
