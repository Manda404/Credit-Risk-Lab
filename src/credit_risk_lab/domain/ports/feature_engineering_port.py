# src/credit_risk_lab/domain/ports/feature_engineering_port.py

from typing import Protocol
from pandas import DataFrame


class FeatureEngineeringPort(Protocol):
    """
    Port (interface) décrivant ce que le domaine attend d'un service
    de feature engineering.

    Rôle :
    - Encapsuler les règles de transformation métier du dataset
      (création de ratios, encodages, interactions, etc.)
    - Permettre plusieurs implémentations (Pandas, Spark, etc.)

    Toute implémentation concrète doit respecter ce contrat.
    """

    def transform(self, df: DataFrame, *, is_training: bool = True) -> DataFrame:
        """
        Applique toutes les transformations de feature engineering
        sur le DataFrame source.

        Parameters
        ----------
        df : DataFrame
            Dataset brut (ou pré-nettoyé) contenant les colonnes d'entrée.
        is_training : bool, default True
            Indique si l'on est en phase d'entraînement (train) ou d'inférence (prod).
            Utile si certaines transformations ne doivent être faites qu'en training.

        Returns
        -------
        DataFrame
            DataFrame enrichi avec les nouvelles features.
        """
        ...
