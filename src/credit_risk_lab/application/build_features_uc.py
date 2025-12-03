# src/credit_risk_lab/application/use_cases/build_features_uc.py

from pandas import DataFrame
from credit_risk_lab.domain.ports.dataset_repository_port import DatasetRepositoryPort
from credit_risk_lab.domain.ports.feature_engineering_port import FeatureEngineeringPort
from credit_risk_lab.shared.logging import setup_logger


class BuildFeaturesUseCase:
    """
    Use Case applicatif :
    - charge un dataset brut via le DatasetRepositoryPort
    - applique le FeatureEngineeringPort
    - retourne le DataFrame enrichi (ou éventuellement le sauvegarde via un autre port)
    """

    def __init__(
        self,
        dataset_repository: DatasetRepositoryPort,
        feature_engineer: FeatureEngineeringPort,
        logger_name: str = "build_features_uc",
    ) -> None:
        self.dataset_repository = dataset_repository
        self.feature_engineer = feature_engineer
        self.logger = setup_logger(logger_name=logger_name)

    def execute(self, *, is_training: bool = True) -> DataFrame:
        self.logger.info("Démarrage du use case BuildFeaturesUseCase...")

        # 1. Charger le dataset brut
        df_raw = self.dataset_repository.load()
        self.logger.info(f"Dataset brut chargé : shape = {df_raw.shape}")

        # 2. Appliquer le feature engineering
        df_features = self.feature_engineer.transform(df_raw, is_training=is_training)
        self.logger.info(f"Dataset enrichi : shape = {df_features.shape}")

        return df_features
