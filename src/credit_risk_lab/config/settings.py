"""
config/settings.py

Ce module définit la configuration globale du projet **Credit Risk Lab**  
(version 0.1.0) — un framework Clean Architecture dédié au scoring
crédit et à l'approbation de prêts.

🎯 Objectifs
------------
Centraliser les paramètres essentiels au projet Loan Approval :
    - métadonnées du projet (nom, version)
    - chemins (datasets, modèles, logs)
    - paramètres MLflow (tracking, registry)
    - configuration des modèles ML (XGB, CatBoost, LGBM)
    - paramètres de monitoring (PSI, KS thresholds)
    - configuration API (si un service de scoring est exposé)
    - chargement des secrets depuis `.env`

Pourquoi une configuration centralisée ?
---------------------------------------
✔ Pour éviter les “chemins en dur”  
✔ Pour faciliter le déploiement (local, cloud, Docker, CI/CD)  
✔ Pour valider automatiquement les dossiers critiques  
✔ Pour garder un objet `settings` unique, propre et réutilisable dans
  tout le projet Credit Risk Lab.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict, field_validator


class Settings(BaseSettings):
    """
    Configuration principale du projet **Credit Risk Lab**.

    Métadonnées incluses :
        - Nom du projet
        - Version du projet

    Exemple d'utilisation :
    -----------------------
        from credit_risk_lab.config.settings import settings

        print(settings.project_name)
        print(settings.project_version)
        print(settings.data_dir)
    """

    # -------------------------------------------------------------------------
    # MÉTADONNÉES DU PROJET
    # -------------------------------------------------------------------------
    project_name: str = Field(
        default="Credit Risk Lab",
        description="Nom officiel du projet de scoring crédit."
    )

    project_version: str = Field(
        default="0.1.0",
        description="Version courante du projet Credit Risk Lab."
    )

    # -------------------------------------------------------------------------
    # CONFIGURATION PYDANTIC
    # -------------------------------------------------------------------------
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # CHEMINS DU PROJET — FIX POUR AVOIR /data, /logs, /models A LA RACINE
    # -------------------------------------------------------------------------

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3],
        description="Racine du projet Credit Risk Lab.",
    )

    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "data",
        description="Dossier principal contenant les datasets Loan Approval.",
    )

    models_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "models",
        description="Dossier où seront sauvegardés les modèles entraînés.",
    )

    logs_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "logs",
        description="Dossier pour les logs de l'application (Loguru).",
    )

    # -------------------------------------------------------------------------
    # MLFLOW : TRACKING + REGISTRY
    # -------------------------------------------------------------------------
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="URI du serveur MLflow pour suivre les entraînements.",
    )

    mlflow_experiment_name: str = Field(
        default="credit_risk_lab_experiment",
        description="Nom de l'expérience MLflow par défaut.",
    )

    mlflow_registry_uri: str | None = Field(
        default=None,
        description="URI du Model Registry MLflow (optionnel).",
    )
    # -------------------------------------------------------------------------
    # PARAMÈTRES DES MODÈLES (HYPERPARAMS PAR DÉFAUT)
    # -------------------------------------------------------------------------

    xgboost_params: dict = Field(
        default={
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "auc",
        },
        description="Hyperparamètres par défaut pour XGBoost.",
    )

    catboost_params: dict = Field(
        default={
            "depth": 6,
            "learning_rate": 0.05,
            "iterations": 300,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": False,
        },
        description="Hyperparamètres par défaut pour CatBoost.",
    )

    lightgbm_params: dict = Field(
        default={
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "objective": "binary",
        },
        description="Hyperparamètres LightGBM par défaut.",
    )

    # -------------------------------------------------------------------------
    # PARAMÈTRES DE DRIFT / MONITORING
    # -------------------------------------------------------------------------

    psi_threshold: float = Field(
        default=0.25,
        description="Seuil PSI au-delà duquel un drift populationnel est déclenché.",
    )

    ks_threshold: float = Field(
        default=0.1,
        description="Seuil KS utilisé pour détecter un drift du modèle.",
    )

    # -------------------------------------------------------------------------
    # API LOCAL (OPTIONNEL) — SCORING SERVICE
    # -------------------------------------------------------------------------
    API_URL: str = Field(
        default="http://localhost",
        env="API_URL",
        description="Adresse de l'API de scoring.",
    )
    API_PORT: int = Field(
        default=8000,
        env="API_PORT",
        gt=0,
        lt=99999,
        description="Port de l'API de scoring.",
    )
    DEBUG: bool = Field(
        default=False,
        env="DEBUG",
        description="Mode debug pour l'API.",
    )

    @property
    def full_api_url(self) -> str:
        return f"{self.API_URL}:{self.API_PORT}"

    # -------------------------------------------------------------------------
    # VALIDATION AUTOMATIQUE DES DOSSIERS
    # -------------------------------------------------------------------------
    @field_validator("data_dir", "logs_dir", "models_dir")
    def ensure_directory_exists(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


# -------------------------------------------------------------------------
# INSTANCE GLOBALE DE CONFIGURATION (SINGLETON LOGIQUE)
# -------------------------------------------------------------------------
settings = Settings()