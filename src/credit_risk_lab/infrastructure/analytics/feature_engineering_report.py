"""Diagnostics for feature-engineering column evolution."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureEngineeringReport:
    """Compare two DataFrames before and after feature engineering."""

    before: pd.DataFrame
    after: pd.DataFrame

    def created_columns(self) -> list[str]:
        """Return columns added by feature engineering."""
        before_columns = set(self.before.columns)
        return [column for column in self.after.columns if column not in before_columns]

    def removed_columns(self) -> list[str]:
        """Return columns removed by feature engineering."""
        after_columns = set(self.after.columns)
        return [column for column in self.before.columns if column not in after_columns]

    def retained_columns(self) -> list[str]:
        """Return columns present before and after feature engineering."""
        after_columns = set(self.after.columns)
        return [column for column in self.before.columns if column in after_columns]

    def changed_columns(self) -> list[str]:
        """Return retained columns whose values changed during transformation."""
        changed = []
        for column in self.retained_columns():
            left = self.before[column].reset_index(drop=True)
            right = self.after[column].reset_index(drop=True)
            if not left.equals(right):
                changed.append(column)
        return changed

    def summary(self) -> pd.DataFrame:
        """Return a compact column-evolution summary."""
        created = self.created_columns()
        removed = self.removed_columns()
        retained = self.retained_columns()
        changed = self.changed_columns()
        return pd.DataFrame(
            [
                {"metric": "columns_before", "value": len(self.before.columns)},
                {"metric": "columns_after", "value": len(self.after.columns)},
                {"metric": "created_columns", "value": len(created)},
                {"metric": "removed_columns", "value": len(removed)},
                {"metric": "retained_columns", "value": len(retained)},
                {"metric": "changed_retained_columns", "value": len(changed)},
                {"metric": "rows_before", "value": len(self.before)},
                {"metric": "rows_after", "value": len(self.after)},
            ]
        )

    def created_columns_frame(self) -> pd.DataFrame:
        """Return metadata for newly created columns."""
        rows = [
            {
                "created_column": column,
                "dtype": str(self.after[column].dtype),
                "missing_rate": float(self.after[column].isna().mean()),
                "unique_values": int(self.after[column].nunique(dropna=False)),
            }
            for column in self.created_columns()
        ]
        return pd.DataFrame(rows)
