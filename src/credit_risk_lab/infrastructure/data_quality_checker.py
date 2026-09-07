"""Object-oriented quality checker for credit-risk datasets."""

import pandas as pd

from credit_risk_lab.domain.entities import LoanSchema, QualityReport
from credit_risk_lab.infrastructure.data_quality import (
    build_quality_report,
    clean_implausible_rows,
    validate_schema,
)


class CreditRiskQualityChecker:
    """Validate and clean a credit-risk training dataset."""

    def __init__(self, schema: LoanSchema | None = None):
        self.schema = schema or LoanSchema()

    def validate(self, frame: pd.DataFrame) -> QualityReport:
        """Validate schema and return structural/business quality diagnostics."""
        return build_quality_report(frame)

    def validate_schema(self, frame: pd.DataFrame) -> None:
        """Raise when the training schema contract is not respected."""
        validate_schema(frame)

    def clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a conservative training sample without implausible rows."""
        return clean_implausible_rows(frame)
