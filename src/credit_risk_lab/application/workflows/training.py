"""Training and persistence workflow."""

from dataclasses import dataclass

import pandas as pd

from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.modeling import (
    BestModelSelector,
    BoostingModelTrainer,
    CreditRiskPreprocessor,
    JoblibModelBundleRepository,
    build_configured_models,
    load_models_config,
    sha256_file,
)

from ._shared import current_git_commit, load_train_dataset, load_validation_dataset


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
    """Train candidates on processed train data and select on validation only."""
    train = load_train_dataset()
    validation = load_validation_dataset()
    target_column = settings.target_column
    x_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    x_validation = validation.drop(columns=[target_column])
    y_validation = validation[target_column]

    trainer = BoostingModelTrainer(random_state=settings.random_state)
    results = trainer.fit(x_train, y_train, x_validation, y_validation)
    validation_metrics = trainer.results_frame(results)
    best_result = BestModelSelector(metric=settings.selection_metric).select(results)
    winner = best_result.model_name
    winner_row = validation_metrics.loc[validation_metrics["model"].eq(winner)].iloc[0]
    preprocessor = CreditRiskPreprocessor.load(settings.preprocessing_artifact_path)
    repository = JoblibModelBundleRepository()
    bundle_path = repository.save(
        settings.model_bundle_path,
        model=best_result.model,
        preprocessor=preprocessor.transformer,
        threshold=float(winner_row["threshold"]),
        metadata={
            "model_name": winner,
            "validation_metrics": winner_row.to_dict(),
            "selection_metric": settings.selection_metric,
            "split_strategy": "raw_train_to_processed_train_validation",
            "training_dataset_sha256": sha256_file(settings.train_path),
            "validation_dataset_sha256": sha256_file(settings.validation_path),
            "preprocessing_artifact_sha256": sha256_file(
                settings.preprocessing_artifact_path
            ),
            "models_config_sha256": sha256_file(settings.models_config_path),
            "git_commit": current_git_commit(),
            "target_definition": "loan_status=1 is the synthetic positive risk class",
        },
    )
    return TrainingWorkflowResult(
        models=inspect_configured_models(),
        validation_metrics=validation_metrics,
        test_metrics=pd.DataFrame(),
        selected_model_name=winner,
        model_bundle_path=bundle_path,
        histories={result.model_name: result.history for result in results},
    )
