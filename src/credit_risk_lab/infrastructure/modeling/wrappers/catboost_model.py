"""CatBoost classifier wrapper."""

from catboost import CatBoostClassifier
from .base import ModelWrapper


class CatBoostWrapper(ModelWrapper):
    """Configure CatBoost from YAML and normalize its log-loss history."""

    def _build_model(self) -> CatBoostClassifier:
        return CatBoostClassifier(random_seed=self.random_state, **self.parameters)

    def fit(self, x_train, y_train, x_validation, y_validation):
        self.model.fit(
            x_train, y_train,
            eval_set=(x_validation, y_validation),
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False,
        )
        raw = self.model.get_evals_result()
        self.history = {
            "train_loss": raw["learn"]["Logloss"],
            "validation_loss": raw["validation"]["Logloss"],
        }
        return self
