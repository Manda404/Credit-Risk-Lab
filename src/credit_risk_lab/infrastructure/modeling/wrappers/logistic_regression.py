"""Interpretable logistic-regression baseline wrapper."""

from sklearn.linear_model import LogisticRegression
from .base import ModelWrapper


class LogisticRegressionWrapper(ModelWrapper):
    """Fit the configured linear baseline without early stopping."""

    def _build_model(self) -> LogisticRegression:
        return LogisticRegression(random_state=self.random_state, **self.parameters)

    def fit(self, x_train, y_train, x_validation, y_validation):
        self.model.fit(x_train, y_train)
        self.history = {}
        return self
