"""Joblib implementation of the model bundle repository port."""

from pathlib import Path

from .artifacts import load_model_bundle, save_model_bundle


class JoblibModelBundleRepository:
    """Persist and load trusted local model bundles with joblib."""

    def load(self, path: Path) -> dict:
        """Load a trusted model bundle."""
        return load_model_bundle(path)

    def save(
        self, path: Path, *, model, preprocessor, threshold: float, metadata: dict
    ) -> Path:
        """Persist a model bundle."""
        return save_model_bundle(
            path,
            model=model,
            preprocessor=preprocessor,
            threshold=threshold,
            metadata=metadata,
        )
