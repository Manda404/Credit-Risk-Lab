"""Explicit preprocessing component for credit-risk notebooks."""

from pathlib import Path

import joblib
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

    def transform_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features and return a DataFrame with stable feature names."""
        matrix = self.transform(features)
        return pd.DataFrame(matrix, columns=self.feature_names_, index=features.index)

    def fit_transform(self, features: pd.DataFrame):
        """Fit preprocessing on training features and return the transformed matrix."""
        self.transformer = build_preprocessor(features)
        matrix = self.transformer.fit_transform(features)
        self.feature_names_ = self.transformer.get_feature_names_out().tolist()
        return matrix

    def fit_transform_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit on training features and return the transformed training DataFrame."""
        matrix = self.fit_transform(features)
        return pd.DataFrame(matrix, columns=self.feature_names_, index=features.index)

    def save(self, path: Path) -> Path:
        """Persist the fitted preprocessing object for validation/test reuse."""
        if self.transformer is None:
            raise RuntimeError("CreditRiskPreprocessor must be fitted before saving")
        output_path = path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        return output_path

    @classmethod
    def load(cls, path: Path) -> "CreditRiskPreprocessor":
        """Load a trusted fitted preprocessing object."""
        preprocessor = joblib.load(path.expanduser().resolve())
        if not isinstance(preprocessor, cls):
            raise ValueError(f"Invalid preprocessor artifact: {path}")
        if preprocessor.transformer is None:
            raise ValueError(f"Unfitted preprocessor artifact: {path}")
        return preprocessor
