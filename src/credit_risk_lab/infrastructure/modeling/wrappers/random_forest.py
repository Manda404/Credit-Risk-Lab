"""Random Forest classifier wrapper."""

from sklearn.ensemble import RandomForestClassifier
from .base import ModelWrapper


class RandomForestWrapper(ModelWrapper):
    """Fit a configured bagging baseline with no iterative loss history.

    Random Forest does not expose an early-stopping validation curve through
    scikit-learn. It still implements the common probability contract and is
    evaluated under exactly the same validation/test protocol as other models.
    """

    def _build_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(random_state=self.random_state, **self.parameters)

    def fit(self, x_train, y_train, x_validation, y_validation):
        self.model.fit(x_train, y_train)
        self.history = {}
        return self
