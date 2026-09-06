import joblib
import pytest

from credit_risk_lab.infrastructure.modeling import (
    JoblibModelBundleRepository,
    load_model_bundle,
)


def test_model_bundle_repository_fails_for_missing_file(tmp_path):
    repository = JoblibModelBundleRepository()
    with pytest.raises(FileNotFoundError):
        repository.load(tmp_path / "missing.joblib")


def test_model_bundle_loader_rejects_corrupted_file(tmp_path):
    path = tmp_path / "corrupted.joblib"
    path.write_bytes(b"not-a-joblib-bundle")
    with pytest.raises(Exception):
        load_model_bundle(path)


def test_model_bundle_loader_rejects_incomplete_contract(tmp_path):
    path = tmp_path / "incomplete.joblib"
    joblib.dump({"model": object()}, path)
    with pytest.raises(ValueError, match="Invalid model bundle"):
        load_model_bundle(path)
