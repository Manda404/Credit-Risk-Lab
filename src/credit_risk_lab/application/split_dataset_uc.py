# src/credit_risk_lab/application/use_cases/split_dataset_uc.py

import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

from credit_risk_lab.domain.ports.dataset_repository_port import DatasetRepositoryPort
from credit_risk_lab.shared.logging import setup_logger


class SplitDatasetUseCase:

    def __init__(self, dataset_repo: DatasetRepositoryPort):
        self.dataset_repo = dataset_repo
        self.logger = setup_logger(name="SplitDatasetUseCase")

    def execute(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        target_column: str = "loan_status",
        save: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # ------------------------------------------------------------
        # 1. Load data via port → Clean Architecture compliant
        # ------------------------------------------------------------
        self.logger.info("Loading dataset via DatasetRepositoryPort...")
        df = self.dataset_repo.load()
        self.logger.info("Dataset loaded successfully.")

        # ------------------------------------------------------------
        # 2. Split (stratified)
        # ------------------------------------------------------------
        self.logger.info(
            f"Splitting dataset: test_size={test_size}, "
            f"random_state={random_state}, target_column={target_column}"
        )

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=df[target_column],
        )

        self.logger.info(
            f"Split completed: train={train_df.shape}, test={test_df.shape}"
        )

        # ------------------------------------------------------------
        # 3. Optional save via the Port (CSV/Parquet/S3 depends on adapter)
        # ------------------------------------------------------------
        if save:
            self.logger.info("Saving split datasets via DatasetRepositoryPort...")
            self.dataset_repo.save(train_df, "train")
            self.dataset_repo.save(test_df, "test")
            self.logger.info("Datasets successfully saved to processed directory.")

        return train_df, test_df
