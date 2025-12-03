"""
feature_functions.py

Ce module contient l'ensemble des fonctions de transformation métier 
utilisées pendant le Feature Engineering du dataset Loan Approval.

Chaque fonction :
- prend un DataFrame en entrée
- applique une transformation spécifique
- retourne le DataFrame modifié
- est 100 % indépendante, donc facilement testable

Objectif : rendre le code lisible, modulaire, réutilisable et documenté.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pandas import DataFrame


# ==========================================================
# 1. INCOME & SOLVENCY FEATURES
# ==========================================================
def add_income_features(df: DataFrame) -> DataFrame:
    """
    Ajoute les variables de solvabilité : DTI, log(income), mensualité estimée,
    ratio paiement/revenu.

    Paramètres
    ----------
    df : DataFrame
        Dataset contenant person_income, loan_amnt, loan_int_rate.

    Retour
    ------
    DataFrame
        Dataset enrichi des variables :
        - dti
        - log_income
        - estimated_monthly_payment
        - payment_to_income
    """
    df["dti"] = df["loan_amnt"] / df["person_income"].replace(0, np.nan)
    df["log_income"] = np.log1p(df["person_income"].clip(lower=0))

    monthly_rate = (df["loan_int_rate"] / 100) / 12
    df["estimated_monthly_payment"] = df["loan_amnt"] * monthly_rate

    df["payment_to_income"] = df["estimated_monthly_payment"] / (
        (df["person_income"] / 12).replace(0, np.nan)
    )

    return df


# ==========================================================
# 2. CREDIT SCORE FEATURES
# ==========================================================
def add_credit_score_features(df: DataFrame) -> DataFrame:
    """
    Ajoute les variables liées au credit score :
    - credit_score_band : segmentation FICO-like
    - norm_credit_score : z-score normalisé

    Paramètres
    ----------
    df : DataFrame
        Dataset contenant credit_score.

    Retour
    ------
    DataFrame
        Dataset enrichi avec credit_score_band & norm_credit_score.
    """
    df["credit_score_band"] = pd.cut(
        df["credit_score"],
        bins=[300, 579, 669, 739, 799, 900],
        labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
        include_lowest=True,
    )

    mean_score = df["credit_score"].mean()
    std_score = df["credit_score"].std(ddof=0) or 1
    df["norm_credit_score"] = (df["credit_score"] - mean_score) / std_score

    return df


# ==========================================================
# 3. AGE & EXPERIENCE FEATURES
# ==========================================================
def add_age_experience_features(df: DataFrame) -> DataFrame:
    """
    Ajoute les variables basées sur l'âge et l'expérience professionnelle :
    - age_group
    - exp_to_age (indicateur de stabilité emploi/âge)
    """
    df["age_group"] = pd.cut(
        df["person_age"],
        bins=[18, 25, 35, 50, 65, 110],
        labels=["18-25", "26-35", "36-50", "51-65", "65+"],
        include_lowest=True,
    )

    df["exp_to_age"] = df["person_emp_exp"] / df["person_age"].replace(0, np.nan)

    return df


# ==========================================================
# 4. LOAN FEATURES
# ==========================================================
def add_loan_features(df: DataFrame) -> DataFrame:
    """
    Ajoute des variables liées au prêt :
    - amnt_int_ratio
    - loan_risk_score
    - loan_intent_risk (mapping métier)
    """
    df["amnt_int_ratio"] = df["loan_amnt"] / df["loan_int_rate"].replace(0, np.nan)
    df["loan_risk_score"] = df["loan_int_rate"] * df["loan_percent_income"]

    intent_map = {
        "MEDICAL": 3,
        "PERSONAL": 2,
        "VENTURE": 2,
        "EDUCATION": 1,
        "HOMEIMPROVEMENT": 1,
        "DEBTCONSOLIDATION": 2,
    }
    df["loan_intent_risk"] = df["loan_intent"].map(intent_map).fillna(1)
    return df


# ==========================================================
# 5. CREDIT HISTORY FEATURES
# ==========================================================
def add_credit_history_features(df: DataFrame) -> DataFrame:
    """
    Ajoute les features liées à l'historique de crédit :
    - credit_hist_to_age
    - credit_hist_category
    """
    df["credit_hist_to_age"] = df["cb_person_cred_hist_length"] / df[
        "person_age"
    ].replace(0, np.nan)

    df["credit_hist_category"] = pd.cut(
        df["cb_person_cred_hist_length"],
        bins=[0, 3, 7, 15, 40],
        labels=["0-3", "4-7", "8-15", "15+"],
        include_lowest=True,
    )
    return df


# ==========================================================
# 6. PREVIOUS DEFAULTS FEATURES
# ==========================================================
def add_default_features(df: DataFrame) -> DataFrame:
    """
    Ajoute les features basées sur les précédents défauts :
    - has_default_before
    - risky_default_score (default × faible credit_score)
    """
    df["has_default_before"] = (
        df["previous_loan_defaults_on_file"].astype(str).str.upper().eq("YES")
    ).astype(int)

    df["risky_default_score"] = df["has_default_before"] * (650 - df["credit_score"])

    return df


# ==========================================================
# 7. CATEGORICAL BUSINESS ENCODINGS
# ==========================================================
def add_business_encoding(df: DataFrame) -> DataFrame:
    """
    Encodage métier des variables catégorielles :
    - home_risk
    - edu_level
    - is_female
    """
    home_map = {"OWN": 2, "MORTGAGE": 1, "RENT": 0, "OTHER": 0}
    df["home_risk"] = (
        df["person_home_ownership"].astype(str).str.upper().map(home_map).fillna(0)
    )

    edu_map = {
        "High School": 0,
        "Associate": 1,
        "Bachelor": 2,
        "Master": 3,
        "Doctorate": 4,
    }
    df["edu_level"] = df["person_education"].map(edu_map).fillna(0)

    df["is_female"] = df["person_gender"].astype(str).str.lower().eq("female").astype(int)

    return df


# ==========================================================
# 8. INTERACTION FEATURES
# ==========================================================
def add_interaction_features(df: DataFrame) -> DataFrame:
    """
    Ajoute des interactions non linéaires entre variables clés :
    - income × interest rate
    - age × credit history
    - loan_amnt × loan_percent_income
    """
    df["income_interest_interaction"] = df["person_income"] * df["loan_int_rate"]
    df["age_credit_interaction"] = df["person_age"] * df["cb_person_cred_hist_length"]
    df["amnt_percent_interaction"] = df["loan_amnt"] * df["loan_percent_income"]

    return df


# ==========================================================
# 9. SANITY CHECKS AND FINAL CLEANING 
# ==========================================================
def sanitize_features(df: DataFrame) -> DataFrame:
    """Nettoie les inf, -inf et applique quelques corrections métiers."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df["dti"] = df["dti"].clip(lower=0)
    df["payment_to_income"] = df["payment_to_income"].clip(lower=0)
    df["exp_to_age"] = df["exp_to_age"].clip(lower=0)
    df["credit_hist_to_age"] = df["credit_hist_to_age"].clip(lower=0)
    return df
