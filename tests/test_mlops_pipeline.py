from types import SimpleNamespace

import pandas as pd

from credit_risk_lab.application import SplitConfig
from credit_risk_lab.application.workflows import CreditRiskMLOpsPipeline
from credit_risk_lab.config.settings import Settings


def build_test_settings(tmp_path, *, minimum_validation_roc_auc: float = 0.98):
    return Settings(
        _env_file=None,
        project_root=tmp_path,
        environment="development",
        raw_data_path="data/raw/loan_data.csv",
        minimum_validation_roc_auc=minimum_validation_roc_auc,
    )


def test_split_manifest_marks_current_split(tmp_path):
    settings = build_test_settings(tmp_path)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    source = pd.DataFrame(
        {
            "feature": range(4),
            "loan_status": [0, 1, 0, 1],
        }
    )
    train = source.iloc[:2].copy()
    test = source.iloc[2:].copy()
    source.to_csv(settings.raw_data_path, index=False)
    train.to_csv(settings.raw_train_path, index=False)
    test.to_csv(settings.raw_test_path, index=False)

    pipeline = CreditRiskMLOpsPipeline(project_settings=settings, optuna_trials=1)
    split_config = SplitConfig(test_size=0.5, random_state=42, stratify=True)
    pipeline._write_split_manifest(source, train, test, split_config)

    assert pipeline._split_manifest_is_current(split_config)

    source.assign(feature=[10, 11, 12, 13]).to_csv(
        settings.raw_data_path,
        index=False,
    )

    assert not pipeline._split_manifest_is_current(split_config)


def test_promotion_gate_rejects_candidate_below_validation_threshold(tmp_path):
    settings = build_test_settings(tmp_path, minimum_validation_roc_auc=0.98)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.candidate_model_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    settings.candidate_model_bundle_path.write_bytes(b"candidate")
    pipeline = CreditRiskMLOpsPipeline(project_settings=settings, optuna_trials=1)
    tuned = SimpleNamespace(metrics={"roc_auc": 0.97})
    evaluation = {"metrics": pd.DataFrame([{"roc_auc": 0.99, "pr_auc": 0.95}])}

    result = pipeline._promotion_gate(tuned=tuned, processed={}, evaluation=evaluation)

    assert not result["approved"]
    assert not settings.model_bundle_path.exists()
    assert result["report"].iloc[0]["promotion_status"] == "rejected"


def test_dockerfile_uses_poetry_lock_for_reproducible_install():
    dockerfile = (Settings().project_root / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY pyproject.toml poetry.lock README.md" in dockerfile
    assert "poetry install --only main" in dockerfile
    assert "pip install ." not in dockerfile
