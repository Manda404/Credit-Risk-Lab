"""XGBoost classifier wrapper."""

from xgboost import XGBClassifier
from .base import ModelWrapper


class XGBoostWrapper(ModelWrapper):
    """Configure XGBoost from YAML and normalize its log-loss history."""

    def _build_model(self) -> XGBClassifier:
        return XGBClassifier(random_state=self.random_state, **self.parameters)

    def fit(self, x_train, y_train, x_validation, y_validation):
        self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
        self.model.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_validation, y_validation)],
            verbose=False,
        )
        raw = self.model.evals_result()
        self.history = {
            "train_loss": raw["validation_0"]["logloss"],
            "validation_loss": raw["validation_1"]["logloss"],
        }
        return self
