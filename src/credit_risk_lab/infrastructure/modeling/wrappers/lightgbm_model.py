"""LightGBM classifier wrapper."""

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from .base import ModelWrapper


class LightGBMWrapper(ModelWrapper):
    """Configure LightGBM from YAML and normalize its log-loss history."""

    def _build_model(self) -> LGBMClassifier:
        return LGBMClassifier(random_state=self.random_state, **self.parameters)

    def fit(self, x_train, y_train, x_validation, y_validation):
        self.model.fit(
            x_train, y_train,
            eval_set=[(x_train, y_train), (x_validation, y_validation)],
            eval_names=["train", "validation"],
            eval_metric="binary_logloss",
            callbacks=[
                early_stopping(self.early_stopping_rounds, verbose=False),
                log_evaluation(0),
            ],
        )
        raw = self.model.evals_result_
        self.history = {
            "train_loss": raw["train"]["binary_logloss"],
            "validation_loss": raw["validation"]["binary_logloss"],
        }
        return self

    def predict_proba(self, features):
        """Preserve generated feature names to avoid LightGBM warnings."""
        if not isinstance(features, pd.DataFrame) and hasattr(self.model, "feature_name_"):
            features = pd.DataFrame(features, columns=self.model.feature_name_)
        return self.model.predict_proba(features)[:, 1]
