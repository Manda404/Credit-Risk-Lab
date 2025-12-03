# src/credit_risk_lab/infrastructure/data_sources/csv_dataset_repository.py

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

from credit_risk_lab.config.settings import settings
from credit_risk_lab.shared.logging import setup_logger
from credit_risk_lab.domain.ports.dataset_repository_port import DatasetRepositoryPort


class CSVDatasetRepository(DatasetRepositoryPort):
    """
    Adapter concret pour charger un dataset CSV.
    Implémente le port DatasetRepositoryPort de la couche Domain.
    """

    def __init__(
        self,
        name_file: str = "loan_data.csv",
        name_folder: str = "raw",  # data/processed, data/raw
        *,
        sep: str = ",",
        encoding: str = "utf-8",
        dtype: Optional[Dict[str, Any]] = None,
        nrows: Optional[int] = None,
        **extra_read_csv_options: Any,
    ) -> None:

        # ------------------------------------------------------------
        # LOGGER — instancié une seule fois
        # ------------------------------------------------------------
        self.logger = setup_logger(name="CSVDatasetRepository")

        # ------------------------------------------------------------
        # Chemin du CSV
        # ------------------------------------------------------------
        self.csv_path = settings.data_dir / name_folder / name_file

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {self.csv_path}")

        self.read_kwargs = {
            "sep": sep,
            "encoding": encoding,
            "dtype": dtype,
            "nrows": nrows,
            **extra_read_csv_options,
        }

        self.logger.debug(
            f"Initialisé avec fichier={self.csv_path} "
            f"options={self.read_kwargs}"
        )

    # ----------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """
        Charge le dataset CSV en DataFrame.
        """
        self.logger.info(f"Chargement du fichier : {self.csv_path}")

        try:
            df = pd.read_csv(self.csv_path, **self.read_kwargs)
        except Exception as e:
            self.logger.error(f"Erreur lors de la lecture CSV : {e}")
            raise

        if df.empty:
            raise ValueError(f"Dataset vide : {self.csv_path}")

        self.logger.info(
            f"Dataset chargé ({df.shape[0]} lignes, {df.shape[1]} colonnes)"
        )

        return df

    # ----------------------------------------------------------------------
    def save(self, data: pd.DataFrame, name_data: str = "train") -> None:
        """
        Sauvegarde un DataFrame en CSV.
        """
        output_path = settings.data_dir / "processed" / f"{name_data}.csv"
        self.logger.info(f"Sauvegarde du fichier : {output_path}")

        try:
            data.to_csv(
                output_path,
                index=False,
                encoding=self.read_kwargs.get("encoding", "utf-8"),
            )
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde CSV : {e}")
            raise

        self.logger.info("Fichier sauvegardé avec succès ...")
