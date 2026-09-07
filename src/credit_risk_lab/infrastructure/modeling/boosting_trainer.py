"""Progressive model training components for notebooks."""

from dataclasses import dataclass

import pandas as pd

from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.evaluation import CreditRiskModelEvaluator
from credit_risk_lab.shared.logging import setup_logger

from .model_factory import build_configured_models


@dataclass(frozen=True)
class CandidateTrainingResult:
    """Validation result for one trained model candidate."""

    model_name: str
    model: object
    threshold: float
    metrics: dict[str, float]
    history: dict[str, list[float]]


class BoostingModelTrainer:
    """Fit configured model candidates on prepared matrices."""

    def __init__(
        self,
        *,
        random_state: int = settings.random_state,
        model_names: list[str] | None = None,
        n_estimators: int | None = None,
        evaluator: CreditRiskModelEvaluator | None = None,
    ):
        self.random_state = random_state
        self.model_names = model_names
        self.n_estimators = n_estimators
        self.evaluator = evaluator or CreditRiskModelEvaluator()

    def fit(
        self,
        x_train,
        y_train,
        x_validation,
        y_validation,
    ) -> list[CandidateTrainingResult]:
        """Train enabled candidates and evaluate them on validation data."""
        overrides = None
        if self.n_estimators is not None:
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
        if self.model_names is not None:
            requested = set(self.model_names)
            models = [model for model in models if model.name in requested]
            if not models:
                raise ValueError(f"No enabled model matches: {sorted(requested)}")
        results = []
        for model in models:
            model.fit(x_train, y_train, x_validation, y_validation)
            probabilities = model.predict_proba(x_validation)
            threshold = self.evaluator.optimal_threshold(y_validation, probabilities)
            metrics = self.evaluator.metrics(y_validation, probabilities, threshold)
            results.append(
                CandidateTrainingResult(
                    model_name=model.name,
                    model=model,
                    threshold=threshold,
                    metrics=metrics,
                    history=model.history,
                )
            )
        return results

    def results_frame(self, results: list[CandidateTrainingResult]) -> pd.DataFrame:
        """Return candidate results sorted by the configured selection metric."""
        rows = [
            {
                "model": result.model_name,
                "threshold": result.threshold,
                **result.metrics,
            }
            for result in results
        ]
        return (
            pd.DataFrame(rows)
            .sort_values(settings.selection_metric, ascending=False)
            .reset_index(drop=True)
        )


class BestModelSelector:
    """Select the best validation candidate according to one metric."""

    def __init__(self, metric: str = settings.selection_metric):
        self.metric = metric
        self.logger = setup_logger(name="BestModelSelector")

    def select(self, results: list[CandidateTrainingResult]) -> CandidateTrainingResult:
        """Return the candidate with the highest configured metric."""
        if not results:
            raise ValueError("Cannot select a model from an empty result list")
        missing = [
            result.model_name for result in results if self.metric not in result.metrics
        ]
        if missing:
            raise ValueError(
                f"Metric {self.metric!r} is missing for candidates: {missing}"
            )
        selected = max(results, key=lambda result: result.metrics[self.metric])
        self.logger.info(
            "Selected model="
            f"{selected.model_name} metric={self.metric} "
            f"value={selected.metrics[self.metric]:.4f} "
            f"threshold={selected.threshold:.4f}"
        )
        return selected
