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
        - estimated_monthly_interest
        - interest_to_income
        - income_per_experience_year
    """
    df["dti"] = df["loan_amnt"] / df["person_income"].replace(0, np.nan)
    df["log_income"] = np.log1p(df["person_income"].clip(lower=0))

    monthly_rate = (df["loan_int_rate"] / 100) / 12
    # Loan term is absent from this teaching dataset, so an amortizing monthly
    # payment cannot be computed. This variable is explicitly an interest-only
    # proxy and must not be presented to decision makers as a real instalment.
    df["estimated_monthly_interest"] = df["loan_amnt"] * monthly_rate

    df["interest_to_income"] = df["estimated_monthly_interest"] / (
        (df["person_income"] / 12).replace(0, np.nan)
    )

    # Career earnings efficiency: income per year of professional experience.
    # Distinct from person_income alone (a young high earner and an older
    # average earner can share the same income but very different values here),
    # which a tree model cannot recover from income or experience in isolation.
    df["income_per_experience_year"] = df["person_income"] / (df["person_emp_exp"] + 1)

    return df


# ==========================================================
# 2. CREDIT SCORE FEATURES
# ==========================================================
def add_credit_score_features(df: DataFrame) -> DataFrame:
    """
    Ajoute les variables liées au credit score :
    - credit_score_band : segmentation FICO-like (bornes fixes, non calculées sur les données)

    Le score brut est déjà standardisé de façon leakage-safe par le
    ``ColumnTransformer`` (fit sur train uniquement, voir ``infrastructure.modeling.preprocessing``).
    Un z-score calculé ici recalculerait moyenne/écart-type sur le DataFrame reçu
    (train+validation+test au moment de l'appel), ce qui constituerait une fuite
    statistique vers le train. Il n'est donc pas dupliqué dans cette étape déterministe.

    Paramètres
    ----------
    df : DataFrame
        Dataset contenant credit_score.

    Retour
    ------
    DataFrame
        Dataset enrichi avec credit_score_band.
    """
    df["credit_score_band"] = pd.cut(
        df["credit_score"],
        bins=[300, 579, 669, 739, 799, 900],
        labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
        include_lowest=True,
    )

    # Interest rate charged per point of credit score: a pricing-efficiency
    # signal distinct from either variable alone — two borrowers with the same
    # score can be priced very differently, and this ratio surfaces that gap.
    df["rate_per_score_point"] = df["loan_int_rate"] / df["credit_score"].replace(0, np.nan)

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
    df["loan_intent_risk"] = df["loan_intent"].map(intent_map)
    return df


# ==========================================================
# 5. CREDIT HISTORY FEATURES
# ==========================================================
def add_credit_history_features(df: DataFrame) -> DataFrame:
    """
    Ajoute les features liées à l'historique de crédit :
    - credit_hist_to_age
    - credit_hist_category
    - age_first_credit
    - emp_credit_hist_gap
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

    # Classic credit-bureau feature: the age at which credit history began.
    # An unusually young value can indicate an inconsistent or thin file.
    df["age_first_credit"] = df["person_age"] - df["cb_person_cred_hist_length"]

    # Whether employment tenure runs ahead of or behind credit tenure — a
    # consistency signal that neither ratio-to-age feature above captures,
    # since both are anchored on age rather than on each other.
    df["emp_credit_hist_gap"] = df["person_emp_exp"] - df["cb_person_cred_hist_length"]

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

    df["risky_default_score"] = df["has_default_before"] * (650 - df["credit_score"]).clip(lower=0)

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
        df["person_home_ownership"].astype(str).str.upper().map(home_map)
    )

    edu_map = {
        "High School": 0,
        "Associate": 1,
        "Bachelor": 2,
        "Master": 3,
        "Doctorate": 4,
    }
    df["edu_level"] = df["person_education"].map(edu_map)

    df["is_female"] = df["person_gender"].astype(str).str.lower().eq("female").astype(int)

    return df


# ==========================================================
# 8. AGGREGATE RISK FLAGS
# ==========================================================
def add_risk_flags(df: DataFrame) -> DataFrame:
    """
    Ajoute un score composite ``risk_flags_count`` : le nombre d'indicateurs
    de risque métier classiques déclenchés simultanément (0 à 5) :
    défaut antérieur, score de crédit subprime (<580), taux d'endettement
    élevé (>40 % du revenu), statut locataire, motif de consolidation de dette.

    Contrairement aux ratios ci-dessus, ce n'est pas une transformation
    monotone d'une seule colonne existante : c'est un décompte de conditions
    indépendantes, donc une information réellement nouvelle pour un modèle
    à arbres (invariant aux transformations monotones d'une variable seule).

    Nécessite que ``has_default_before`` ait déjà été calculé
    (voir ``add_default_features``), appelé avant cette fonction dans le pipeline.
    """
    df["risk_flags_count"] = (
        df["has_default_before"]
        + (df["credit_score"] < 580).astype(int)
        + (df["loan_percent_income"] > 0.40).astype(int)
        + df["person_home_ownership"].astype(str).str.upper().eq("RENT").astype(int)
        + df["loan_intent"].astype(str).str.upper().eq("DEBTCONSOLIDATION").astype(int)
    )
    return df


# ==========================================================
# 9. INTERACTION FEATURES
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
# 10. SANITY CHECKS AND FINAL CLEANING
# ==========================================================
def sanitize_features(df: DataFrame) -> DataFrame:
    """Nettoie les inf, -inf et applique quelques corrections métiers."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df["dti"] = df["dti"].clip(lower=0)
    df["interest_to_income"] = df["interest_to_income"].clip(lower=0)
    df["exp_to_age"] = df["exp_to_age"].clip(lower=0)
    df["credit_hist_to_age"] = df["credit_hist_to_age"].clip(lower=0)
    df["income_per_experience_year"] = df["income_per_experience_year"].clip(lower=0)
    df["age_first_credit"] = df["age_first_credit"].clip(lower=0)
    return df
