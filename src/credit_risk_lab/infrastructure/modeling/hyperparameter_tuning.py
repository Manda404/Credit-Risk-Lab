"""Small validation-only hyperparameter tuning utilities."""

from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd
from sklearn.model_selection import ParameterGrid

from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.evaluation import CreditRiskModelEvaluator
from credit_risk_lab.shared.logging import setup_logger

from .model_config import ModelsConfig, load_models_config
from .model_factory import build_configured_models
from .wrappers import CatBoostWrapper


@dataclass(frozen=True)
class TuningCandidateResult:
    """Validation result for one model/hyperparameter candidate."""

    model_name: str
    parameters: dict[str, Any]
    model: object
    threshold: float
    metrics: dict[str, float]
    history: dict[str, list[float]]


class ModelHyperparameterTuner:
    """Tune only selected baseline models on the validation partition."""

    def __init__(
        self,
        *,
        random_state: int = settings.random_state,
        metric: str = "roc_auc",
        evaluator: CreditRiskModelEvaluator | None = None,
        config: ModelsConfig | None = None,
    ):
        self.random_state = random_state
        self.metric = metric
        self.evaluator = evaluator or CreditRiskModelEvaluator()
        self.config = config or load_models_config()
        self.logger = setup_logger(name="ModelHyperparameterTuner")

    def tune(
        self,
        *,
        model_names: list[str],
        parameter_grids: dict[str, list[dict[str, Any]]],
        x_train,
        y_train,
        x_validation,
        y_validation,
    ) -> list[TuningCandidateResult]:
        """Return validation results for configured grids and selected models."""
        if not model_names:
            raise ValueError("model_names must contain at least one model")
        results = []
        model_keys = self._model_keys_by_display_name()
        for model_name in model_names:
            if model_name not in model_keys:
                raise ValueError(f"Unknown configured model display name: {model_name}")
            key = model_keys[model_name]
            for parameters in self._parameter_combinations(
                parameter_grids.get(model_name, [{}])
            ):
                overrides = {key: parameters}
                model = self._build_single_model(model_name, overrides)
                self.logger.info(f"Tuning model={model_name} parameters={parameters}")
                model.fit(x_train, y_train, x_validation, y_validation)
                probabilities = model.predict_proba(x_validation)
                threshold = self.evaluator.optimal_threshold(
                    y_validation,
                    probabilities,
                )
                metrics = self.evaluator.metrics(
                    y_validation,
                    probabilities,
                    threshold,
                )
                results.append(
                    TuningCandidateResult(
                        model_name=model.name,
                        parameters=parameters,
                        model=model,
                        threshold=threshold,
                        metrics=metrics,
                        history=model.history,
                    )
                )
        return results

    def results_frame(self, results: list[TuningCandidateResult]) -> pd.DataFrame:
        """Return tuning results sorted by the configured selection metric."""
        rows = [
            {
                "model": result.model_name,
                "parameters": result.parameters,
                "threshold": result.threshold,
                **result.metrics,
            }
            for result in results
        ]
        return (
            pd.DataFrame(rows)
            .sort_values(self.metric, ascending=False)
            .reset_index(drop=True)
        )

    def select_best(
        self,
        results: list[TuningCandidateResult],
    ) -> TuningCandidateResult:
        """Select the best tuned candidate."""
        if not results:
            raise ValueError("Cannot select from empty tuning results")
        missing = [
            result.model_name for result in results if self.metric not in result.metrics
        ]
        if missing:
            raise ValueError(
                f"Metric {self.metric!r} is missing for tuned candidates: {missing}"
            )
        selected = max(results, key=lambda result: result.metrics[self.metric])
        self.logger.info(
            "Selected tuned model="
            f"{selected.model_name} metric={self.metric} "
            f"value={selected.metrics[self.metric]:.4f} "
            f"threshold={selected.threshold:.4f} "
            f"parameters={selected.parameters}"
        )
        return selected

    def _model_keys_by_display_name(self) -> dict[str, str]:
        return {
            definition.display_name: key
            for key, definition in self.config.models.items()
            if definition.enabled
        }

    def _build_single_model(self, model_name: str, overrides: dict[str, dict]):
        models = build_configured_models(
            random_state=self.random_state,
            config=self.config,
            parameter_overrides=overrides,
        )
        matching = [model for model in models if model.name == model_name]
        if not matching:
            raise ValueError(f"Enabled model not found: {model_name}")
        return matching[0]

    def _parameter_combinations(
        self,
        grids: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        combinations = []
        for grid in grids:
            combinations.extend(dict(parameters) for parameters in ParameterGrid(grid))
        return combinations or [{}]


@dataclass(frozen=True)
class OptunaTuningResult:
    """Best Optuna optimization result and validation diagnostics."""

    model_name: str
    best_parameters: dict[str, Any]
    best_value: float
    threshold: float
    metrics: dict[str, float]
    model: object
    study: optuna.Study
    trials: pd.DataFrame


class CatBoostOptunaTuner:
    """Optimize CatBoost hyperparameters on validation data only."""

    def __init__(
        self,
        *,
        random_state: int = settings.random_state,
        metric: str = settings.selection_metric,
        n_trials: int = 25,
        timeout: int | None = None,
        evaluator: CreditRiskModelEvaluator | None = None,
    ):
        if CatBoostWrapper is None:
            raise RuntimeError("CatBoost must be installed before tuning CatBoost")
        self.random_state = random_state
        self.metric = metric
        self.n_trials = n_trials
        self.timeout = timeout
        self.evaluator = evaluator or CreditRiskModelEvaluator()
        self.logger = setup_logger(name="CatBoostOptunaTuner")

    def tune(self, x_train, y_train, x_validation, y_validation) -> OptunaTuningResult:
        """Run Optuna and refit the best CatBoost candidate."""
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            study_name="catboost_validation_tuning",
        )
        study.optimize(
            lambda trial: self._objective(
                trial,
                x_train,
                y_train,
                x_validation,
                y_validation,
            ),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )
        best_parameters = self._runtime_parameters(study.best_params)
        best_model = self._build_model(best_parameters)
        best_model.fit(x_train, y_train, x_validation, y_validation)
        probabilities = best_model.predict_proba(x_validation)
        threshold = self.evaluator.optimal_threshold(y_validation, probabilities)
        metrics = self.evaluator.metrics(y_validation, probabilities, threshold)
        trials = self.trials_frame(study)
        self.logger.info(
            "Selected CatBoost Optuna model "
            f"metric={self.metric} value={metrics[self.metric]:.4f} "
            f"threshold={threshold:.4f} parameters={study.best_params}"
        )
        return OptunaTuningResult(
            model_name=best_model.name,
            best_parameters=study.best_params,
            best_value=float(study.best_value),
            threshold=threshold,
            metrics=metrics,
            model=best_model,
            study=study,
            trials=trials,
        )

    def trials_frame(self, study: optuna.Study) -> pd.DataFrame:
        """Return Optuna trials sorted by validation objective."""
        rows = []
        for trial in study.trials:
            row = {
                "trial": trial.number,
                "value": trial.value,
                "state": trial.state.name,
            }
            row.update(trial.params)
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .sort_values("value", ascending=False, na_position="last")
            .reset_index(drop=True)
        )

    def _objective(
        self,
        trial: optuna.Trial,
        x_train,
        y_train,
        x_validation,
        y_validation,
    ) -> float:
        parameters = self._runtime_parameters(self._suggest_parameters(trial))
        model = self._build_model(parameters)
        model.fit(x_train, y_train, x_validation, y_validation)
        probabilities = model.predict_proba(x_validation)
        threshold = self.evaluator.optimal_threshold(y_validation, probabilities)
        metrics = self.evaluator.metrics(y_validation, probabilities, threshold)
        return float(metrics[self.metric])

    def _suggest_parameters(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "iterations": trial.suggest_int("iterations", 200, 1200),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.005,
                0.3,
                log=True,
            ),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
            "random_strength": trial.suggest_float(
                "random_strength", 0.1, 20.0, log=True
            ),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 80),
            "rsm": trial.suggest_float("rsm", 0.5, 1.0),
            "auto_class_weights": trial.suggest_categorical(
                "auto_class_weights",
                [None, "Balanced", "SqrtBalanced"],
            ),
        }

    def _runtime_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            **parameters,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "verbose": False,
            "allow_writing_files": False,
        }

    def _build_model(self, parameters: dict[str, Any]):
        return CatBoostWrapper(
            name="CatBoost",
            parameters=parameters,
            random_state=self.random_state,
            early_stopping_rounds=30,
        )
