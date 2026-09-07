"""Model-specific feature importance utilities."""

import numpy as np
import pandas as pd


class CatBoostFeatureImportanceAnalyzer:
    """Extract normalized CatBoost feature importances for notebook diagnostics."""

    def __init__(self, model, feature_names: list[str]):
        self.model = getattr(model, "model", model)
        self.feature_names = list(feature_names)

    def importance_frame(self, top_n: int = 20) -> pd.DataFrame:
        """Return top feature importances as percentages."""
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model does not expose CatBoost feature_importances_")
        importances = np.asarray(self.model.feature_importances_, dtype=float)
        if len(importances) != len(self.feature_names):
            raise ValueError(
                "Feature importance length does not match provided feature names"
            )
        total = importances.sum()
        if total <= 0:
            normalized = np.zeros_like(importances)
        else:
            normalized = 100 * importances / total
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": importances,
                    "importance_pct": normalized,
                }
            )
            .sort_values("importance_pct", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
