"""Repository ports for data and model artifacts."""

from pathlib import Path
from typing import Protocol

import pandas as pd


class DatasetRepository(Protocol):
    """Load and save tabular datasets without exposing storage details."""

    def load(self) -> pd.DataFrame: ...

    def save(
        self, data: pd.DataFrame, path: Path, *, encoding: str = "utf-8"
    ) -> Path: ...


class ModelBundleRepository(Protocol):
    """Persist and load trusted model bundles."""

    def load(self, path: Path) -> dict: ...

    def save(
        self, path: Path, *, model, preprocessor, threshold: float, metadata: dict
    ) -> Path: ...
