"""Common wrapper contract independent from any ML library."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class ModelWrapper(ABC):
    """Expose probabilities and normalized training history for every model."""

    name: str

    def __init__(
        self,
        *,
        name: str,
        parameters: dict[str, Any],
        random_state: int,
        early_stopping_rounds: int | None = None,
    ) -> None:
        self.name = name
        self.parameters = parameters.copy()
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.model = self._build_model()
        self.history: dict[str, list[float]] = {}

    @abstractmethod
    def _build_model(self) -> Any:
        """Instantiate the underlying estimator from injected parameters."""

    @abstractmethod
    def fit(self, x_train, y_train, x_validation, y_validation) -> "ModelWrapper":
        """Fit the estimator and return this wrapper."""

    def predict_proba(self, features) -> np.ndarray:
        """Return the positive-class probability."""
        return self.model.predict_proba(features)[:, 1]

    def predict(self, features, threshold: float = 0.5) -> np.ndarray:
        """Convert probabilities to binary risk decisions."""
        return (self.predict_proba(features) >= threshold).astype(int)
