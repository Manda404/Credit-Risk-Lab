"""Column diagnostics for data-quality notebooks."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class OutlierConfig:
    """Configuration for IQR-based outlier detection."""

    lower_quantile: float = 0.25
    upper_quantile: float = 0.75
    iqr_multiplier: float = 1.5


class ColumnDiagnostics:
    """Profile numeric outliers and categorical quality signals."""

    def __init__(self, frame: pd.DataFrame, target_column: str | None = None):
        if frame is None or frame.empty:
            raise ValueError("ColumnDiagnostics requires a non-empty DataFrame")
        self.frame = frame
        self.target_column = target_column

    def numeric_features(self) -> list[str]:
        """Return numeric feature columns, excluding the target when present."""
        columns = self.frame.select_dtypes(include="number").columns.tolist()
        return [column for column in columns if column != self.target_column]

    def categorical_features(self) -> list[str]:
        """Return categorical feature columns."""
        return self.frame.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

    def numeric_outlier_report(
        self, config: OutlierConfig | None = None
    ) -> pd.DataFrame:
        """Return IQR outlier boundaries and counts for numeric features."""
        config = config or OutlierConfig()
        rows = []
        for column in self.numeric_features():
            series = self.frame[column].dropna()
            if series.empty:
                continue
            q1 = float(series.quantile(config.lower_quantile))
            q3 = float(series.quantile(config.upper_quantile))
            iqr = q3 - q1
            lower_bound = q1 - config.iqr_multiplier * iqr
            upper_bound = q3 + config.iqr_multiplier * iqr
            outliers = series.lt(lower_bound) | series.gt(upper_bound)
            rows.append(
                {
                    "feature": column,
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                    "q1": q1,
                    "q3": q3,
                    "iqr": float(iqr),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outlier_count": int(outliers.sum()),
                    "outlier_rate": float(outliers.mean()),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values(["outlier_rate", "outlier_count"], ascending=False)
            .reset_index(drop=True)
        )

    def categorical_profile(self, rare_threshold: float = 0.01) -> pd.DataFrame:
        """Return cardinality and rare-category diagnostics."""
        if not 0 <= rare_threshold <= 1:
            raise ValueError("rare_threshold must be between 0 and 1")
        rows = []
        for column in self.categorical_features():
            series = self.frame[column].astype("string").fillna("<missing>")
            frequencies = series.value_counts(normalize=True)
            counts = series.value_counts()
            rare = frequencies[frequencies < rare_threshold]
            rows.append(
                {
                    "feature": column,
                    "cardinality": int(series.nunique(dropna=False)),
                    "missing_count": int(series.eq("<missing>").sum()),
                    "top_category": str(counts.index[0]),
                    "top_count": int(counts.iloc[0]),
                    "top_rate": float(frequencies.iloc[0]),
                    "rare_category_count": int(len(rare)),
                    "rare_categories": rare.index.astype(str).tolist(),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values(["cardinality", "rare_category_count"], ascending=False)
            .reset_index(drop=True)
        )
