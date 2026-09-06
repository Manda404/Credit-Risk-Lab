"""Dataset schema and business-plausibility checks."""

from dataclasses import dataclass

import pandas as pd


REQUIRED_COLUMNS = {
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "previous_loan_defaults_on_file", "loan_status",
}


@dataclass(frozen=True)
class QualityReport:
    """Compact result returned by the quality gate."""

    rows: int
    columns: int
    duplicate_rows: int
    missing_values: int
    invalid_age_rows: int
    invalid_experience_rows: int

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


def validate_schema(df: pd.DataFrame) -> None:
    """Raise a clear error when required columns or target values are invalid."""
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    target_values = set(df["loan_status"].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"loan_status must be binary, received {target_values}")


def build_quality_report(df: pd.DataFrame) -> QualityReport:
    """Measure structural quality without silently changing source records."""
    validate_schema(df)
    return QualityReport(
        rows=len(df), columns=len(df.columns), duplicate_rows=int(df.duplicated().sum()),
        missing_values=int(df.isna().sum().sum()),
        invalid_age_rows=int((~df["person_age"].between(18, 100)).sum()),
        invalid_experience_rows=int(((df["person_emp_exp"] < 0) | (df["person_emp_exp"] > 80) | (df["person_emp_exp"] > df["person_age"] - 14)).sum()),
    )


def clean_implausible_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a documented conservative sample for model development."""
    validate_schema(df)
    mask = df["person_age"].between(18, 100) & df["person_emp_exp"].ge(0)
    mask &= df["person_emp_exp"].le(80) & df["person_emp_exp"].le(df["person_age"] - 14)
    return df.loc[mask].drop_duplicates().reset_index(drop=True)
