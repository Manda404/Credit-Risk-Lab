"""Modeling domain entities."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelCandidateSpec:
    """Versioned model candidate chosen by configuration."""

    key: str
    display_name: str
    enabled: bool
    parameters: dict[str, Any]
    early_stopping_rounds: int | None = None


@dataclass(frozen=True)
class ModelTrainingSummary:
    """Minimal domain summary of model selection."""

    selected_model_name: str
    selection_metric: str
    threshold: float
