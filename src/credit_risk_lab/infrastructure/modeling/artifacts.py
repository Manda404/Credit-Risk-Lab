"""Portable local persistence for fitted preprocessing and model wrappers."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import platform

import joblib


def sha256_file(path: Path) -> str:
    """Return a stable content fingerprint without loading a file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    """Capture runtime versions needed to diagnose artifact incompatibility."""
    packages = ["pandas", "scikit-learn", "xgboost", "catboost", "lightgbm", "joblib"]
    versions = {name: importlib.metadata.version(name) for name in packages}
    return {"python": platform.python_version(), **versions}


def save_model_bundle(path: Path, *, model, preprocessor, threshold: float, metadata: dict) -> Path:
    """Persist an inference bundle with schema and reproducibility metadata.

    Joblib/pickle artifacts must only be loaded from trusted storage. They are
    not a secure interchange format and can execute code during deserialization.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_versions": runtime_versions(),
        "feature_schema": [str(c) for c in getattr(preprocessor, "feature_names_in_", [])],
        **metadata,
    }
    joblib.dump({"model": model, "preprocessor": preprocessor, "threshold": threshold, "metadata": enriched}, path)
    return path


def load_model_bundle(path: Path) -> dict:
    """Load a trusted bundle and validate its minimum inference contract."""
    bundle = joblib.load(path)
    required = {"model", "preprocessor", "threshold", "metadata"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(f"Invalid model bundle; expected keys {sorted(required)}")
    if not 0 <= float(bundle["threshold"]) <= 1:
        raise ValueError("Invalid decision threshold in model bundle")
    return bundle
