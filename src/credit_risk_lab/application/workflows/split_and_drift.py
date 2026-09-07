"""Development split and drift workflow."""

from dataclasses import dataclass

import pandas as pd

from credit_risk_lab.application import create_deployment_split
from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.analytics import DriftAnalyzer
from credit_risk_lab.infrastructure.data_sources import CSVDatasetRepository
from credit_risk_lab.infrastructure.feature_engineering import LoanFeatureEngineer
from credit_risk_lab.infrastructure.modeling import CreditRiskPreprocessor
from credit_risk_lab.infrastructure.visualization import plot_drift_summary

from ._shared import load_raw_train_dataset


@dataclass(frozen=True)
class SplitAndDriftResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    processed_train: pd.DataFrame
    processed_validation: pd.DataFrame
    split_summary: pd.DataFrame
    drift_report: pd.DataFrame
    preprocessor_path: object


def run_split_and_drift_workflow() -> SplitAndDriftResult:
    """Create raw train/validation diagnostics and persist transformed datasets."""
    split = create_deployment_split(
        load_raw_train_dataset(),
        test_size=settings.validation_size,
    )
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

    feature_engineer = LoanFeatureEngineer()
    featured_train = feature_engineer.transform(train)
    featured_validation = feature_engineer.transform(validation)
    sensitive_columns = [
        column for column in settings.sensitive_columns if column in featured_train
    ]
    drop_columns = [settings.target_column, *sensitive_columns]
    y_train = featured_train[settings.target_column].reset_index(drop=True)
    y_validation = featured_validation[settings.target_column].reset_index(drop=True)
    x_train = featured_train.drop(columns=drop_columns)
    x_validation = featured_validation.drop(columns=drop_columns)

    preprocessor = CreditRiskPreprocessor()
    processed_train_features = preprocessor.fit_transform_frame(x_train)
    processed_validation_features = preprocessor.transform_frame(x_validation)
    processed_train = processed_train_features.assign(
        **{settings.target_column: y_train.to_numpy()}
    )
    processed_validation = processed_validation_features.assign(
        **{settings.target_column: y_validation.to_numpy()}
    )
    CSVDatasetRepository.save(processed_train, settings.train_path)
    CSVDatasetRepository.save(processed_validation, settings.validation_path)
    preprocessor_path = preprocessor.save(settings.preprocessing_artifact_path)

    return SplitAndDriftResult(
        train=train,
        validation=validation,
        processed_train=processed_train,
        processed_validation=processed_validation,
        split_summary=split_summary,
        drift_report=drift_report,
        preprocessor_path=preprocessor_path,
    )
