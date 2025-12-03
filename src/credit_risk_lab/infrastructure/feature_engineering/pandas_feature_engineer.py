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
        df = add_income_features(df)

        # 3. Credit score features
        df = add_credit_score_features(df)

        # 4. Age & employment features
        df = add_age_experience_features(df)

        # 5. Loan-related features
        df = add_loan_features(df)

        # 6. Credit history features
        df = add_credit_history_features(df)

        # 7. Previous defaults features
        df = add_default_features(df)

        # 8. Encodage métier
        df = add_business_encoding(df)

        # 9. Interactions
        df = add_interaction_features(df)

        # 10. Nettoyage final
        df = sanitize_features(df)

        self._log_shape(df, "Sortie")
        return df

    # ------------------------------------------------------------------
    # Méthodes internes : validations & logging
    # ------------------------------------------------------------------
    def _validate_input_columns(self, df: DataFrame) -> None:
        """
        Vérifie que toutes les colonnes nécessaires au feature engineering
        sont bien présentes dans le DataFrame.
        """
        required_cols = [
            "person_age",
            "person_gender",
            "person_education",
            "person_income",
            "person_emp_exp",
            "person_home_ownership",
            "loan_amnt",
            "loan_intent",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            "credit_score",
            "previous_loan_defaults_on_file",
        ]

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            msg = f"Colonnes manquantes pour le feature engineering : {missing}"
            self.logger.error(msg)
            raise ValueError(msg)

    def _log_shape(self, df: DataFrame, step: str) -> None:
        self.logger.info(f"[FeatureEngineering] {step} - shape = {df.shape}")
