"""Explicitly configured CSV loading and saving."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd

from credit_risk_lab.shared.logging import setup_logger


@dataclass(frozen=True)
class CSVDataSourceConfig:
    """Everything required to read one CSV source.

    Keeping the path in this object makes data lineage visible at the call site:
    ``CSVDatasetRepository(config).load()`` never guesses a folder or filename.
    Extra options are forwarded to :func:`pandas.read_csv`.
    """

    path: Path
    sep: str = ","
    encoding: str = "utf-8"
    dtype: dict[str, Any] | None = None
    nrows: int | None = None
    read_options: dict[str, Any] = field(default_factory=dict)

    def read_kwargs(self) -> dict[str, Any]:
        """Return clean keyword arguments accepted by ``pandas.read_csv``."""
        options: dict[str, Any] = {"sep": self.sep, "encoding": self.encoding}
        if self.dtype is not None:
            options["dtype"] = self.dtype
        if self.nrows is not None:
            options["nrows"] = self.nrows
        return {**options, **self.read_options}


class CSVDatasetRepository:
    """Read and write CSV files using an explicit source configuration."""

    def __init__(self, config: CSVDataSourceConfig) -> None:
        self.config = config
        self.csv_path = config.path.expanduser().resolve()
        self.logger = setup_logger(name="CSVDatasetRepository")

    def load(self) -> pd.DataFrame:
        """Load the configured CSV and fail clearly for missing or empty data."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {self.csv_path}")
        if not self.csv_path.is_file():
            raise ValueError(f"La source CSV n'est pas un fichier : {self.csv_path}")
        self.logger.info(f"Chargement du fichier : {self.csv_path}")
        try:
            frame = pd.read_csv(self.csv_path, **self.config.read_kwargs())
        except Exception as exc:
            self.logger.error(f"Erreur lors de la lecture CSV : {exc}")
            raise
        if frame.empty:
            raise ValueError(f"Dataset vide : {self.csv_path}")
        self.logger.info(f"Dataset chargé ({frame.shape[0]} lignes, {frame.shape[1]} colonnes)")
        return frame

    @staticmethod
    def save(data: pd.DataFrame, path: Path, *, encoding: str = "utf-8") -> Path:
        """Persist a DataFrame to the explicit destination and return its path."""
        output_path = path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False, encoding=encoding)
        return output_path
