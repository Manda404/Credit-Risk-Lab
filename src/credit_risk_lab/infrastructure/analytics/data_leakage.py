"""Data leakage diagnostics for split validation."""

import pandas as pd


class DataLeakageAuditor:
    """Detect simple leakage signals between isolated tabular splits."""

    def __init__(self, target_column: str):
        self.target_column = target_column

    def row_overlap_report(
        self,
        train: pd.DataFrame,
        holdout: pd.DataFrame,
        *,
        holdout_name: str = "holdout",
    ) -> pd.DataFrame:
        """Return exact row-overlap diagnostics between two splits."""
        train_hashes = pd.util.hash_pandas_object(train, index=False)
        holdout_hashes = pd.util.hash_pandas_object(holdout, index=False)
        overlap_count = int(holdout_hashes.isin(set(train_hashes)).sum())
        return pd.DataFrame(
            [
                {
                    "check": "exact_row_overlap",
                    "reference": "train",
                    "holdout": holdout_name,
                    "train_rows": len(train),
                    "holdout_rows": len(holdout),
                    "overlap_count": overlap_count,
                    "overlap_rate": overlap_count / len(holdout) if len(holdout) else 0,
                    "passed": overlap_count == 0,
                }
            ]
        )

    def target_leakage_report(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return whether the target column is present in a feature matrix."""
        target_present = self.target_column in features.columns
        return pd.DataFrame(
            [
                {
                    "check": "target_absent_from_features",
                    "target_column": self.target_column,
                    "target_present": target_present,
                    "passed": not target_present,
                }
            ]
        )
