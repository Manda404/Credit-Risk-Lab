"""Factory translating validated YAML definitions into model wrappers."""

from pathlib import Path
from typing import Any

from .model_config import ModelsConfig, load_models_config
from .wrappers import (
    CatBoostWrapper,
    LightGBMWrapper,
    LogisticRegressionWrapper,
    RandomForestWrapper,
    ModelWrapper,
    XGBoostWrapper,
)


WRAPPER_TYPES: dict[str, type[ModelWrapper]] = {
    "logistic_regression": LogisticRegressionWrapper,
    "random_forest": RandomForestWrapper,
    "xgboost": XGBoostWrapper,
    "catboost": CatBoostWrapper,
    "lightgbm": LightGBMWrapper,
}


def build_configured_models(
    *,
    random_state: int,
    config: ModelsConfig | None = None,
    config_path: Path | None = None,
    parameter_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[ModelWrapper]:
    """Instantiate every enabled candidate with parameters injected from YAML.

    ``parameter_overrides`` is intended for tests and controlled experiments.
    Production defaults remain versioned in ``configs/models.yaml``.
    """
    loaded = config or load_models_config(config_path)
    overrides = parameter_overrides or {}
    unknown = sorted(set(loaded.models).difference(WRAPPER_TYPES))
    if unknown:
        raise ValueError(f"No wrapper registered for configured models: {unknown}")
    wrappers = []
    for key, definition in loaded.models.items():
        if not definition.enabled:
            continue
        parameters = {**definition.parameters, **overrides.get(key, {})}
        wrappers.append(WRAPPER_TYPES[key](
            name=definition.display_name,
            parameters=parameters,
            random_state=random_state,
            early_stopping_rounds=definition.early_stopping_rounds,
        ))
    return wrappers
