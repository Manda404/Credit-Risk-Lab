"""Pandas adapter for dataset schema and business-plausibility checks."""

import pandas as pd

from credit_risk_lab.domain.entities import QualityReport
from credit_risk_lab.domain.services import (
    REQUIRED_COLUMNS,
    invalid_age_mask,
    invalid_experience_mask,
    validate_training_schema,
)


def validate_schema(df: pd.DataFrame) -> None:
    """Raise a clear error when required columns or target values are invalid."""
    validate_training_schema(df)


def build_quality_report(df: pd.DataFrame) -> QualityReport:
    """Measure structural quality without silently changing source records."""
    validate_schema(df)
    return QualityReport(
        rows=len(df),
        columns=len(df.columns),
        duplicate_rows=int(df.duplicated().sum()),
        missing_values=int(df.isna().sum().sum()),
        invalid_age_rows=int(invalid_age_mask(df).sum()),
        invalid_experience_rows=int(invalid_experience_mask(df).sum()),
    )


def clean_implausible_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a documented conservative sample for model development."""
    validate_schema(df)
    mask = ~invalid_age_mask(df) & ~invalid_experience_mask(df)
    return df.loc[mask].drop_duplicates().reset_index(drop=True)
