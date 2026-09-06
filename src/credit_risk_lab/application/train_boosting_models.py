"""End-to-end use case for fair comparison of three boosting models."""

from dataclasses import dataclass

import pandas as pd
from credit_risk_lab.config.settings import settings
from credit_risk_lab.application.dataset_splitting import (
    three_way_stratified_split,
    three_way_temporal_split,
)
from credit_risk_lab.infrastructure.evaluation import (
    classification_metrics,
    decile_table,
    find_optimal_threshold,
)
from credit_risk_lab.infrastructure.modeling import (
    build_configured_models,
    build_preprocessor,
)
from credit_risk_lab.shared.logging import setup_logger


@dataclass
class TrainingResult:
    """Artifacts from model development with test isolation made explicit."""

    metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    selected_model_name: str
    histories: dict[str, dict[str, list[float]]]
    models: dict[str, object]
    preprocessor: object
    deciles: dict[str, pd.DataFrame]
    split_summary: pd.DataFrame
    sensitive_test: pd.DataFrame
    split_strategy: str


class TrainBoostingModelsUseCase:
    """Split once, preprocess without leakage, train, tune thresholds, and test."""

    def __init__(
        self,
        random_state: int = settings.random_state,
        n_estimators: int | None = None,
        sensitive_columns: tuple[str, ...] = settings.sensitive_columns,
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.sensitive_columns = sensitive_columns
        self.logger = setup_logger(name="TrainBoostingModelsUseCase")

    def execute(
        self,
        frame: pd.DataFrame,
        target_column: str = settings.target_column,
        *,
        split_strategy: str = settings.split_strategy,
        date_column: str | None = settings.decision_date_column,
        group_column: str | None = settings.borrower_id_column,
    ) -> TrainingResult:
        """Train candidates, select on validation, and test only the winner.

        ``random_experimental`` exists for undated synthetic data. Production
        credit development must pass ``split_strategy="temporal"`` plus the
        application-time date column (and preferably a borrower identifier).
        """
        if split_strategy == "temporal":
            if not date_column:
                raise ValueError(
                    "A decision date column is mandatory for temporal validation"
                )
            split = three_way_temporal_split(
                frame,
                date_column=date_column,
                target_column=target_column,
                group_column=group_column,
            )
        elif split_strategy == "random_experimental":
            self.logger.warning(
                "Using random split: results are educational, not production evidence"
            )
            split = three_way_stratified_split(frame, target_column, self.random_state)
        else:
            raise ValueError(
                "split_strategy must be 'temporal' or 'random_experimental'"
            )
        x_train, x_validation, x_test = split.x_train, split.x_validation, split.x_test
        y_train, y_validation, y_test = split.y_train, split.y_validation, split.y_test
        split_summary = pd.DataFrame(
            [
                {
                    "split": "train",
                    "rows": len(y_train),
                    "positive_rate": y_train.mean(),
                },
                {
                    "split": "validation",
                    "rows": len(y_validation),
                    "positive_rate": y_validation.mean(),
                },
                {"split": "test", "rows": len(y_test), "positive_rate": y_test.mean()},
            ]
        )
        self.logger.info(
            f"Dataset split completed: {split_summary.to_dict(orient='records')}"
        )

        present_sensitive = [c for c in self.sensitive_columns if c in x_test.columns]
        sensitive_test = x_test[present_sensitive].copy()
        if present_sensitive:
            self.logger.info(
                f"Excluding sensitive columns from training features: {present_sensitive}"
            )
            x_train = x_train.drop(columns=present_sensitive)
            x_validation = x_validation.drop(columns=present_sensitive)
            x_test = x_test.drop(columns=present_sensitive)

        preprocessor = build_preprocessor(x_train)
        x_train_t = preprocessor.fit_transform(x_train)
        x_validation_t = preprocessor.transform(x_validation)
        x_test_t = preprocessor.transform(x_test)

        rows, histories, fitted, validation_probabilities_by_model = [], {}, {}, {}
        overrides = None
        if self.n_estimators is not None:
            # Test/experiment override only. Normal runs use configs/models.yaml.
            overrides = {
                "random_forest": {"n_estimators": self.n_estimators},
                "xgboost": {"n_estimators": self.n_estimators},
                "catboost": {"iterations": self.n_estimators},
                "lightgbm": {"n_estimators": self.n_estimators},
            }
        models = build_configured_models(
            random_state=self.random_state,
            parameter_overrides=overrides,
        )
        for model in models:
            self.logger.info(f"Training {model.name}")
            model.fit(x_train_t, y_train, x_validation_t, y_validation)
            validation_probabilities = model.predict_proba(x_validation_t)
            threshold = find_optimal_threshold(y_validation, validation_probabilities)
            metrics = classification_metrics(
                y_validation, validation_probabilities, threshold
            )
            rows.append({"model": model.name, "threshold": threshold, **metrics})
            histories[model.name] = model.history
            fitted[model.name] = model
            validation_probabilities_by_model[model.name] = validation_probabilities
            self.logger.info(
                f"{model.name} validation ROC-AUC={metrics['roc_auc']:.4f}"
            )
        validation_metrics = (
            pd.DataFrame(rows)
            .sort_values(settings.selection_metric, ascending=False)
            .reset_index(drop=True)
        )
        selected_name = str(validation_metrics.iloc[0]["model"])
        selected_threshold = float(validation_metrics.iloc[0]["threshold"])
        test_probabilities = fitted[selected_name].predict_proba(x_test_t)
        final_test_metrics = classification_metrics(
            y_test, test_probabilities, selected_threshold
        )
        test_metrics = pd.DataFrame(
            [
                {
                    "model": selected_name,
                    "threshold": selected_threshold,
                    **final_test_metrics,
                }
            ]
        )
        deciles = {selected_name: decile_table(y_test, test_probabilities)}
        self.logger.info(
            f"Selected {selected_name} on validation; final test ROC-AUC={final_test_metrics['roc_auc']:.4f}"
        )
        return TrainingResult(
            validation_metrics,
            test_metrics,
            selected_name,
            histories,
            fitted,
            preprocessor,
            deciles,
            split_summary,
            sensitive_test,
            split_strategy,
        )
