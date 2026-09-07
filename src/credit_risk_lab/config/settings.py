"""Typed, side-effect-free project configuration.

Every file location used by the project (raw data, processed datasets,
model bundle, reports, logs) is defined once here and exposed as a `*_path`
property. Nothing else in the codebase — application code, infrastructure,
scripts, or notebooks — should rebuild a project path by hand
(e.g. ``ROOT / "data" / "processed" / "modeling_dataset.csv"``); it must
import `settings` and read the corresponding property instead.

Values are loaded, in increasing priority, from:
1. the field defaults below,
2. ``configs/settings.yaml`` (versioned, shared across the team),
3. ``configs/environments/{environment}.yaml`` (environment overrides),
4. ``.env`` (local overrides),
5. ``CRL_``-prefixed environment variables (deployment overrides).
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from dotenv import dotenv_values

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_YAML = _PROJECT_ROOT / "configs" / "settings.yaml"
_ENV_CONFIG_DIR = _PROJECT_ROOT / "configs" / "environments"


def _active_environment() -> str:
    """Resolve the active environment before Pydantic sources are built."""
    dotenv = dotenv_values(_PROJECT_ROOT / ".env")
    return (
        os.getenv("CRL_ENVIRONMENT")
        or os.getenv("CRL_ENV")
        or dotenv.get("CRL_ENVIRONMENT")
        or dotenv.get("CRL_ENV")
        or "development"
    )


def _environment_yaml_path() -> Path:
    """Return the YAML override file for the active environment."""
    return _ENV_CONFIG_DIR / f"{_active_environment()}.yaml"


class Settings(BaseSettings):
    """Runtime settings loaded from configs/settings.yaml and CRL-prefixed environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CRL_",
        extra="ignore",
        case_sensitive=False,
        yaml_file=_CONFIG_YAML,
    )

    project_name: str = "Credit Risk Lab"
    project_version: str = "0.2.0"
    environment: Literal["development", "staging", "production"] = "development"
    project_root: Path = Field(default_factory=lambda: _PROJECT_ROOT)

    target_column: str = "loan_status"
    random_state: int = 42
    test_size: float = 0.20
    validation_size: float = 0.20

    selection_metric: str = "roc_auc"
    split_strategy: str = "random_experimental"
    decision_date_column: str | None = None
    borrower_id_column: str | None = None
    decision_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    sensitive_columns: tuple[str, ...] = ("person_gender", "is_female")

    log_level: str = "INFO"
    log_file: str = "credit_risk_lab.log"
    processed_subdir: Path = Path("processed")
    models_subdir: Path = Path("models")
    reports_subdir: Path = Path("reports")
    logs_subdir: Path = Path("logs")
    mlflow_tracking_uri: str = "file:./mlruns/development"
    mlflow_experiment_name: str = "credit-risk-lab-development"

    raw_data_path_config: Path = Field(
        default=Path("data/raw/loan_data.csv"),
        validation_alias="raw_data_path",
    )
    raw_data_sep: str = ","
    raw_data_encoding: str = "utf-8"
    raw_train_file: str = "train.csv"
    raw_test_file: str = "test.csv"
    modeling_dataset_file: str = "modeling_dataset.csv"
    train_file: str = "train.csv"
    validation_file: str = "validation.csv"
    test_file: str = "test.csv"
    preprocessing_artifact_file: str = "credit_risk_preprocessor.joblib"

    model_bundle_file: str = "best_boosting_model.joblib"
    metrics_report_file: str = "boosting_model_metrics.csv"
    models_config_file: str = "models.yaml"
    drift_report_file: str = "deployment_split_drift.csv"
    drift_summary_plot_file: str = "deployment_split_drift.html"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(
                settings_cls,
                yaml_file=_environment_yaml_path(),
            ),
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    # ------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / self.processed_subdir

    @property
    def models_dir(self) -> Path:
        return self.project_root / self.models_subdir

    @property
    def logs_dir(self) -> Path:
        return self.project_root / self.logs_subdir

    @property
    def reports_dir(self) -> Path:
        return self.project_root / self.reports_subdir

    # ------------------------------------------------------------
    # Concrete file paths
    # ------------------------------------------------------------
    @property
    def raw_data_path(self) -> Path:
        """Resolve the configured raw CSV path against the project root."""
        path = self.raw_data_path_config
        return path if path.is_absolute() else self.project_root / path

    @property
    def raw_train_path(self) -> Path:
        return self.raw_dir / self.raw_train_file

    @property
    def raw_test_path(self) -> Path:
        return self.raw_dir / self.raw_test_file

    @property
    def modeling_dataset_path(self) -> Path:
        return self.processed_dir / self.modeling_dataset_file

    @property
    def train_path(self) -> Path:
        return self.processed_dir / self.train_file

    @property
    def validation_path(self) -> Path:
        return self.processed_dir / self.validation_file

    @property
    def test_path(self) -> Path:
        return self.processed_dir / self.test_file

    @property
    def preprocessing_artifact_path(self) -> Path:
        """Fitted preprocessing pipeline used to transform validation and test data."""
        return self.processed_dir / self.preprocessing_artifact_file

    @property
    def model_bundle_path(self) -> Path:
        return self.models_dir / self.model_bundle_file

    @property
    def metrics_report_path(self) -> Path:
        return self.reports_dir / self.metrics_report_file

    @property
    def models_config_path(self) -> Path:
        """Versioned YAML containing model activation and hyperparameters."""
        return self.project_root / "configs" / self.models_config_file

    @property
    def drift_report_path(self) -> Path:
        """CSV containing feature-level train/test drift metrics."""
        return self.reports_dir / self.drift_report_file

    @property
    def drift_summary_plot_path(self) -> Path:
        """Interactive HTML summary of train/test PSI values."""
        return self.reports_dir / self.drift_summary_plot_file

    @property
    def log_file_path(self) -> Path:
        return self.logs_dir / self.log_file

    def decile_report_path(self, model_name: str) -> Path:
        """Path of the per-model lift/gains report, e.g. reports/lightgbm_deciles.csv."""
        return self.reports_dir / f"{model_name.lower()}_deciles.csv"


settings = Settings()
