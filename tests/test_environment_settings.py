from credit_risk_lab.config.settings import Settings


def build_settings(monkeypatch, environment: str) -> Settings:
    monkeypatch.setenv("CRL_ENVIRONMENT", environment)
    return Settings(_env_file=None)


def test_development_environment_keeps_local_lab_paths(monkeypatch):
    settings = build_settings(monkeypatch, "development")

    assert settings.environment == "development"
    assert settings.raw_train_path.parent == settings.raw_dir
    assert settings.raw_test_path.parent == settings.raw_dir
    assert settings.raw_train_path != settings.raw_test_path
    assert settings.validation_path.parent == settings.processed_dir
    assert settings.processed_dir.name == "processed"
    assert settings.models_dir.name == "models"
    assert settings.reports_dir.name == "reports"
    assert settings.mlflow_experiment_name == "credit-risk-lab-development"


def test_staging_and_production_artifacts_are_isolated(monkeypatch):
    staging = build_settings(monkeypatch, "staging")
    production = build_settings(monkeypatch, "production")

    assert staging.environment == "staging"
    assert production.environment == "production"
    assert staging.train_path != production.train_path
    assert staging.model_bundle_path != production.model_bundle_path
    assert staging.reports_dir != production.reports_dir
    assert staging.mlflow_tracking_uri != production.mlflow_tracking_uri


def test_environment_variables_keep_highest_priority(monkeypatch):
    monkeypatch.setenv("CRL_ENVIRONMENT", "staging")
    monkeypatch.setenv("CRL_DECISION_THRESHOLD", "0.33")

    settings = Settings(_env_file=None)

    assert settings.environment == "staging"
    assert settings.decision_threshold == 0.33
