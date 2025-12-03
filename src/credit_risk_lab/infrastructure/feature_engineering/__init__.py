# src/credit_risk_lab/infrastructure/feature_engineering/pandas_feature_engineer.py

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame

from credit_risk_lab.shared.logging import setup_logger
from credit_risk_lab.domain.ports.feature_engineering_port import FeatureEngineeringPort
from credit_risk_lab.infrastructure.feature_engineering.feature_functions import (
    add_income_features,
    add_credit_score_features,
    add_age_experience_features,
    add_loan_features,
    add_credit_history_features,
    add_default_features,
    add_business_encoding,
    add_interaction_features,
    sanitize_features,
)


class LoanFeatureEngineer(FeatureEngineeringPort):
    """
    Implémentation concrète du Feature Engineering pour le cas d'usage :
    Loan Approval Classification Dataset.

    Règles métier intégrées :
    - ratios de solvabilité (DTI, payment_to_income, etc.)
    - engineering du credit score
    - expérience vs âge
    - structure du prêt et risque
    - historique de crédit
    - encodage métier des catégorielles (home_ownership, education, etc.)
    - interactions simples entre variables importantes
    """

    def __init__(self, logger_name: str = "loan_feature_engineer") -> None:
        self.logger = setup_logger(logger_name)

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def transform(self, df: DataFrame, *, is_training: bool = True) -> DataFrame:
        """
        Applique toutes les transformations de feature engineering
        de manière orchestrée et modulaire.
        """

        if df is None or df.empty:
            self.logger.warning("Feature engineering appelé avec un DataFrame vide.")
            return df

        df = df.copy()

        self._log_shape(df, "Entrée")

        # 1. Vérifications minimales des colonnes attendues
        self._validate_input_columns(df)

        # 2. Income & solvency features
        df = self._create_income_features(df)

        # 3. Credit score features
        df = self._create_credit_score_features(df)

        # 4. Age & employment features
        df = self._create_age_experience_features(df)

        # 5. Loan-related features
        df = self._create_loan_features(df)

        # 6. Credit history features
        df = self._create_credit_history_features(df)

        # 7. Previous defaults features
        df = self._create_previous_default_features(df)

        # 8. Encodage métier des catégorielles
        df = self._encode_categorical_features(df)

        # 9. Features d'interaction
        df = self._create_interaction_features(df)

        # 10. Nettoyage final / sanity checks
        df = self._sanitize_values(df)

        self._log_shape(df, "Sortie")

        return df