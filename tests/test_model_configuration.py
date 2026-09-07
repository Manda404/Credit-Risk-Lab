from pathlib import Path
import pytest
from pydantic import ValidationError

from credit_risk_lab.infrastructure.modeling import (
    build_configured_models,
    load_models_config,
)


def test_models_are_instantiated_from_versioned_yaml():
    config = load_models_config()
    models = build_configured_models(random_state=7, config=config)
    assert [model.name for model in models] == [
        "LogisticRegression",
        "RandomForest",
        "XGBoost",
        "CatBoost",
    ]
    assert config.models["lightgbm"].enabled is False
    assert config.models["catboost"].enabled is True
    xgboost = next(model for model in models if model.name == "XGBoost")
    assert xgboost.parameters["eval_metric"] == "logloss"
    assert xgboost.model.get_params()["random_state"] == 7


def test_disabled_model_is_not_instantiated(tmp_path: Path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n"
        "  logistic_regression:\n"
        "    enabled: true\n"
        "    display_name: Baseline\n"
        "    parameters: {max_iter: 50}\n"
        "  lightgbm:\n"
        "    enabled: false\n"
        "    display_name: LightGBM\n"
        "    parameters: {}\n",
        encoding="utf-8",
    )
    models = build_configured_models(random_state=42, config_path=path)
    assert [model.name for model in models] == ["Baseline"]


def test_unknown_model_configuration_fails_loudly(tmp_path: Path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n  unknown:\n    enabled: true\n    display_name: Unknown\n    parameters: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No wrapper registered"):
        build_configured_models(random_state=42, config_path=path)


def test_model_configuration_rejects_unknown_fields(tmp_path: Path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n  xgboost:\n    enabled: true\n    display_name: XGBoost\n"
        "    typo_parameter_block: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_models_config(path)
