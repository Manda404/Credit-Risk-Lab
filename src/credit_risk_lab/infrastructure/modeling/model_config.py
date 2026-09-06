"""Typed loading of model candidates and hyperparameters from YAML."""

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from credit_risk_lab.config.settings import settings


class ModelDefinition(BaseModel):
    """Configuration for one model wrapper."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    display_name: str = Field(min_length=1)
    early_stopping_rounds: int | None = Field(default=None, ge=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    """Validated collection of model definitions keyed by factory name."""

    model_config = ConfigDict(extra="forbid")
    models: dict[str, ModelDefinition]

    @model_validator(mode="after")
    def require_enabled_model(self) -> "ModelsConfig":
        """Reject a configuration that would train no candidate."""
        if not any(model.enabled for model in self.models.values()):
            raise ValueError("At least one model must be enabled")
        return self


def load_models_config(path: Path | None = None) -> ModelsConfig:
    """Load and strictly validate the central ``configs/models.yaml`` file."""
    config_path = path or settings.models_config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Model configuration not found: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        content = yaml.safe_load(stream)
    return ModelsConfig.model_validate(content)
