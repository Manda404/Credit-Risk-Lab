"""Dataset inspection helpers used by notebooks and checks."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetSummary:
    """Compact overview of a tabular dataset."""

    rows: int
    columns: int
    duplicate_rows: int
    memory_mb: float


class DatasetInspector:
    """Expose small, explicit dataset diagnostics."""

    def __init__(self, frame: pd.DataFrame):
        if frame is None or frame.empty:
            raise ValueError("DatasetInspector requires a non-empty DataFrame")
        self.frame = frame

    def summary(self) -> DatasetSummary:
        """Return row, column, duplicate, and memory information."""
        return DatasetSummary(
            rows=len(self.frame),
            columns=len(self.frame.columns),
            duplicate_rows=int(self.frame.duplicated().sum()),
            memory_mb=round(
                float(self.frame.memory_usage(deep=True).sum() / 1024**2), 4
            ),
        )

    def schema(self) -> pd.DataFrame:
        """Return one row per column with dtype, missingness, and cardinality."""
        rows = []
        for column in self.frame.columns:
            series = self.frame[column]
            rows.append(
                {
                    "column": column,
                    "dtype": str(series.dtype),
                    "missing": int(series.isna().sum()),
                    "missing_rate": float(series.isna().mean()),
                    "cardinality": int(series.nunique(dropna=True)),
                }
            )
        return pd.DataFrame(rows)

    def column_summary(self, sample_size: int = 2) -> pd.DataFrame:
        """Return a compact per-column profile with representative examples."""
        if sample_size < 1:
            raise ValueError("sample_size must be at least 1")
        profile = self.schema()
        profile["total_rows"] = len(self.frame)
        profile["all_values_unique"] = profile["cardinality"].eq(len(self.frame))
        profile["examples"] = [
            self.frame[column]
            .dropna()
            .sample(
                min(sample_size, self.frame[column].dropna().shape[0]), random_state=0
            )
            .tolist()
            for column in profile["column"]
        ]
        return profile

    def missing_values(self) -> pd.DataFrame:
        """Return columns with missing values, sorted by rate descending."""
        missing = self.schema()
        return (
            missing.loc[missing["missing"] > 0, ["column", "missing", "missing_rate"]]
            .sort_values("missing_rate", ascending=False)
            .reset_index(drop=True)
        )

    def target_distribution(self, target_column: str) -> pd.DataFrame:
        """Return target counts and rates."""
        if target_column not in self.frame.columns:
            raise ValueError(f"Target column not found: {target_column}")
        counts = (
            self.frame[target_column]
            .value_counts(dropna=False)
            .rename_axis("class")
            .reset_index(name="rows")
        )
        counts["rate"] = counts["rows"] / len(self.frame)
        return counts

    def numeric_profile(self) -> pd.DataFrame:
        """Return standard descriptive statistics for numeric columns."""
        return self.frame.select_dtypes("number").describe().T

    def categorical_cardinality(self) -> pd.DataFrame:
        """Return cardinality for categorical-like columns."""
        categorical = self.frame.select_dtypes(include=["object", "category", "bool"])
        return (
            pd.DataFrame(
                {
                    "column": categorical.columns,
                    "cardinality": [
                        int(categorical[column].nunique(dropna=True))
                        for column in categorical.columns
                    ],
                }
            )
            .sort_values("cardinality", ascending=False)
            .reset_index(drop=True)
        )
