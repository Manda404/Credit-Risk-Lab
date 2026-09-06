"""Quality reporting entities."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityReport:
    """Structural and business-plausibility quality result."""

    rows: int
    columns: int
    duplicate_rows: int
    missing_values: int
    invalid_age_rows: int
    invalid_experience_rows: int

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()
