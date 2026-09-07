"""Explicit dataset splitting component for notebooks and runners."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk_lab.config.settings import settings


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for a two-way holdout split."""

    test_size: float = 0.10
    random_state: int = settings.random_state
    stratify: bool = True
    target_column: str = settings.target_column


class DatasetSplitter:
    """Split a dataset into train and holdout partitions."""

    def __init__(self, config: SplitConfig | None = None):
        self.config = config or SplitConfig()

    def split(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return train and holdout DataFrames with reset indexes."""
        if not 0 < self.config.test_size < 1:
            raise ValueError("test_size must be strictly between 0 and 1")
        stratify = None
        if self.config.stratify:
            if self.config.target_column not in frame.columns:
                raise ValueError(
                    f"Cannot stratify without target column: {self.config.target_column}"
                )
            stratify = frame[self.config.target_column]
        train, holdout = train_test_split(
            frame,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify,
        )
        return train.reset_index(drop=True), holdout.reset_index(drop=True)

    def summary(self, train: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
        """Return row counts and target rates for the split."""
        target = self.config.target_column
        rows = []
        for name, part in [("train", train), ("holdout", holdout)]:
            row = {"sample": name, "rows": len(part)}
            if target in part.columns:
                row["positive_rate"] = float(part[target].mean())
            rows.append(row)
        return pd.DataFrame(rows)
