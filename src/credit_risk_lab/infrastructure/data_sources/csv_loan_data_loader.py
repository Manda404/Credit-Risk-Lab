"""Named CSV loader for raw loan application datasets."""

from pathlib import Path

import pandas as pd

from .csv_dataset_repository import CSVDataSourceConfig, CSVDatasetRepository


class CsvLoanDataLoader:
    """Load a loan dataset from an explicit CSV path."""

    def __init__(self, path: Path, *, sep: str = ",", encoding: str = "utf-8"):
        self.config = CSVDataSourceConfig(path=path, sep=sep, encoding=encoding)

    def load(self) -> pd.DataFrame:
        """Return the configured dataset as a DataFrame."""
        return CSVDatasetRepository(self.config).load()
