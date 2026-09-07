"""Development split and drift workflow."""

from dataclasses import dataclass

import pandas as pd

from credit_risk_lab.application import create_deployment_split
from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.analytics import DriftAnalyzer
from credit_risk_lab.infrastructure.data_sources import CSVDatasetRepository
from credit_risk_lab.infrastructure.visualization import plot_drift_summary

from ._shared import load_raw_train_dataset


@dataclass(frozen=True)
class SplitAndDriftResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    split_summary: pd.DataFrame
    drift_report: pd.DataFrame


def run_split_and_drift_workflow() -> SplitAndDriftResult:
    """Create a train/validation split from raw_train and compute drift diagnostics."""
    split = create_deployment_split(
        load_raw_train_dataset(),
        test_size=settings.validation_size,
        train_path=settings.train_path,
        test_path=settings.validation_path,
    )
    CSVDatasetRepository.save(split.train, settings.train_path)
    CSVDatasetRepository.save(split.test, settings.validation_path)
    train = split.train
    validation = split.test
    split_summary = pd.DataFrame(
        {
            "sample": ["train", "validation"],
            "rows": [split.train_rows, split.test_rows],
            "positive_rate": [
                train[settings.target_column].mean(),
                validation[settings.target_column].mean(),
            ],
        }
    )
    features = [column for column in train.columns if column != settings.target_column]
    drift_report = DriftAnalyzer(bins=10).report_frame(
        train, validation, features=features
    )
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    drift_report.to_csv(settings.drift_report_path, index=False)
    plot_drift_summary(drift_report).write_html(
        settings.drift_summary_plot_path, include_plotlyjs="cdn"
    )
    return SplitAndDriftResult(
        train=train,
        validation=validation,
        split_summary=split_summary,
        drift_report=drift_report,
    )
