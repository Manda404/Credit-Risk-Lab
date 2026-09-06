"""Source quality workflow."""

from dataclasses import dataclass

import pandas as pd

from credit_risk_lab.config.settings import settings
from credit_risk_lab.domain.entities import QualityReport
from credit_risk_lab.infrastructure.data_quality import (
    build_quality_report,
    clean_implausible_rows,
)

from ._shared import load_raw_dataset


@dataclass(frozen=True)
class SourceQualityResult:
    raw: pd.DataFrame
    clean: pd.DataFrame
    quality_report: QualityReport
    summary: pd.DataFrame


def run_source_quality_workflow() -> SourceQualityResult:
    """Load the raw source and return quality diagnostics."""
    raw = load_raw_dataset()
    clean = clean_implausible_rows(raw)
    quality = build_quality_report(raw)
    summary = pd.DataFrame(
        [
            {
                **quality.as_dict(),
                "configured_path": str(settings.raw_data_path_config),
                "resolved_path": str(settings.raw_data_path),
                "clean_rows": len(clean),
                "removed_rows": len(raw) - len(clean),
            }
        ]
    )
    return SourceQualityResult(
        raw=raw, clean=clean, quality_report=quality, summary=summary
    )
