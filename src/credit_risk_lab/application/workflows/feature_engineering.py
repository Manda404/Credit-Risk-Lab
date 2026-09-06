"""Feature engineering workflow."""

from dataclasses import dataclass

import pandas as pd

from credit_risk_lab.application import three_way_stratified_split
from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.feature_engineering import LoanFeatureEngineer
from credit_risk_lab.infrastructure.modeling import build_preprocessor

from ._shared import load_train_dataset


@dataclass(frozen=True)
class FeatureEngineeringResult:
    raw_train: pd.DataFrame
    engineered_train: pd.DataFrame
    new_features: pd.DataFrame
    preprocessing_summary: dict


def run_feature_engineering_workflow() -> FeatureEngineeringResult:
    """Run deterministic features and demonstrate train-only learned preprocessing."""
    raw_train = load_train_dataset()
    engineered_train = LoanFeatureEngineer().transform(raw_train)
    created = [column for column in engineered_train if column not in raw_train]
    split = three_way_stratified_split(engineered_train)
    sensitive = [
        column for column in settings.sensitive_columns if column in split.x_train
    ]
    x_train = split.x_train.drop(columns=sensitive)
    x_validation = split.x_validation.drop(columns=sensitive)
    preprocessor = build_preprocessor(x_train)
    train_matrix = preprocessor.fit_transform(x_train)
    validation_matrix = preprocessor.transform(x_validation)
    summary = {
        "train_matrix": train_matrix.shape,
        "validation_matrix": validation_matrix.shape,
        "output_features": preprocessor.get_feature_names_out()[:20].tolist(),
    }
    new_features = pd.DataFrame(
        {
            "feature": created,
            "missing": engineered_train[created].isna().sum().values,
        }
    )
    return FeatureEngineeringResult(raw_train, engineered_train, new_features, summary)
