"""External holdout and drift workflow."""

from dataclasses import dataclass

import pandas as pd

from credit_risk_lab.application import create_deployment_split
from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.analytics import DriftAnalyzer
from credit_risk_lab.infrastructure.data_sources import CSVDatasetRepository
from credit_risk_lab.infrastructure.visualization import plot_drift_summary

from ._shared import load_raw_dataset


@dataclass(frozen=True)
class SplitAndDriftResult:
    train: pd.DataFrame
    external_test: pd.DataFrame
    split_summary: pd.DataFrame
    drift_report: pd.DataFrame


def run_split_and_drift_workflow() -> SplitAndDriftResult:
    """Create the 90/10 raw holdout, persist it, and compute drift diagnostics."""
    split = create_deployment_split(load_raw_dataset(), test_size=0.10)
    CSVDatasetRepository.save(split.train, settings.train_path)
    CSVDatasetRepository.save(split.test, settings.test_path)
    train = split.train
    external_test = split.test
    split_summary = pd.DataFrame(
        {
            "sample": ["train", "external_test"],
            "rows": [split.train_rows, split.test_rows],
            "positive_rate": [
                train[settings.target_column].mean(),
                external_test[settings.target_column].mean(),
            ],
        }
    )
    features = [column for column in train.columns if column != settings.target_column]
    drift_report = DriftAnalyzer(bins=10).report_frame(
        train, external_test, features=features
    )
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    drift_report.to_csv(settings.drift_report_path, index=False)
    plot_drift_summary(drift_report).write_html(
        settings.drift_summary_plot_path, include_plotlyjs="cdn"
    )
    return SplitAndDriftResult(
        train=train,
        external_test=external_test,
        split_summary=split_summary,
        drift_report=drift_report,
    )
