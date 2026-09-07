"""Explicit preprocessing component for credit-risk notebooks."""

import pandas as pd

from .preprocessing import build_preprocessor


class CreditRiskPreprocessor:
    """Fit and apply the package preprocessing pipeline without leakage."""

    def __init__(self):
        self.transformer = None
        self.feature_names_: list[str] = []

    def fit(self, features: pd.DataFrame) -> "CreditRiskPreprocessor":
        """Fit preprocessing on training features only."""
        self.transformer = build_preprocessor(features)
        self.transformer.fit(features)
        self.feature_names_ = self.transformer.get_feature_names_out().tolist()
        return self

    def transform(self, features: pd.DataFrame):
        """Transform features using the fitted preprocessing pipeline."""
        if self.transformer is None:
            raise RuntimeError("CreditRiskPreprocessor must be fitted before transform")
        return self.transformer.transform(features)

    def fit_transform(self, features: pd.DataFrame):
        """Fit preprocessing on training features and return the transformed matrix."""
        self.transformer = build_preprocessor(features)
        matrix = self.transformer.fit_transform(features)
        self.feature_names_ = self.transformer.get_feature_names_out().tolist()
        return matrix
