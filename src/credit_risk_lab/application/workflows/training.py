"""Training and persistence workflow."""

from dataclasses import dataclass

import pandas as pd

from credit_risk_lab.application import TrainBoostingModelsUseCase
from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.feature_engineering import LoanFeatureEngineer
from credit_risk_lab.infrastructure.modeling import (
    JoblibModelBundleRepository,
    build_configured_models,
    load_models_config,
    sha256_file,
)

from ._shared import current_git_commit, load_train_dataset


@dataclass(frozen=True)
class TrainingWorkflowResult:
    models: pd.DataFrame
    validation_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    selected_model_name: str
    model_bundle_path: object
    histories: dict[str, dict[str, list[float]]]


def inspect_configured_models() -> pd.DataFrame:
    """Return enabled configured model candidates for display."""
    models = build_configured_models(
        random_state=settings.random_state,
        config=load_models_config(),
    )
    return pd.DataFrame(
        [
            {
                "model": model.name,
                "parameters": model.parameters,
                "early_stopping": model.early_stopping_rounds,
            }
            for model in models
        ]
    )


def run_training_workflow() -> TrainingWorkflowResult:
    """Engineer the training partition, train candidates, and persist the winner."""
    train_raw = load_train_dataset()
    train_features = LoanFeatureEngineer().transform(train_raw)
    result = TrainBoostingModelsUseCase().execute(train_features)
    winner = result.selected_model_name
    winner_row = result.test_metrics.iloc[0]
    repository = JoblibModelBundleRepository()
    bundle_path = repository.save(
        settings.model_bundle_path,
        model=result.models[winner],
        preprocessor=result.preprocessor,
        threshold=float(winner_row["threshold"]),
        metadata={
            "model_name": winner,
            "test_metrics": winner_row.to_dict(),
            "selection_metric": settings.selection_metric,
            "split_strategy": result.split_strategy,
            "training_dataset_sha256": sha256_file(settings.train_path),
            "external_test_dataset_sha256": sha256_file(settings.raw_test_path),
            "models_config_sha256": sha256_file(settings.models_config_path),
            "git_commit": current_git_commit(),
            "target_definition": "loan_status=1 is the synthetic positive risk class",
        },
    )
    return TrainingWorkflowResult(
        models=inspect_configured_models(),
        validation_metrics=result.metrics,
        test_metrics=result.test_metrics,
        selected_model_name=winner,
        model_bundle_path=bundle_path,
        histories=result.histories,
    )
