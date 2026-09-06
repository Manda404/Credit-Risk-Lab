"""Service ports used by application use cases."""

from typing import Protocol

import numpy as np
import pandas as pd


class FeatureEngineer(Protocol):
    """Deterministic feature transformer for raw loan applications."""

    def transform(
        self, frame: pd.DataFrame, *, is_training: bool = True
    ) -> pd.DataFrame: ...


class PreprocessorFactory(Protocol):
    """Factory for learned preprocessing pipelines."""

    def __call__(self, features: pd.DataFrame): ...


class ModelCandidate(Protocol):
    """Trainable probabilistic model candidate."""

    name: str
    history: dict[str, list[float]]
    parameters: dict
    early_stopping_rounds: int | None

    def fit(self, x_train, y_train, x_validation, y_validation): ...

    def predict_proba(self, features) -> np.ndarray: ...


class MetricsCalculator(Protocol):
    """Classification metric calculator."""

    def classification_metrics(
        self, y_true, probabilities: np.ndarray, threshold: float
    ) -> dict[str, float]: ...
