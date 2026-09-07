"""Shared workflow helpers kept out of notebooks and scripts."""

import subprocess

import pandas as pd

from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.data_sources import (
    CSVDataSourceConfig,
    CSVDatasetRepository,
)


def load_raw_dataset() -> pd.DataFrame:
    """Load the configured raw source."""
    source = CSVDataSourceConfig(
        path=settings.raw_data_path,
        sep=settings.raw_data_sep,
        encoding=settings.raw_data_encoding,
    )
    return CSVDatasetRepository(source).load()


def load_train_dataset() -> pd.DataFrame:
    """Load the persisted processed training partition."""
    return CSVDatasetRepository(CSVDataSourceConfig(path=settings.train_path)).load()


def load_validation_dataset() -> pd.DataFrame:
    """Load the persisted processed validation partition."""
    return CSVDatasetRepository(
        CSVDataSourceConfig(path=settings.validation_path)
    ).load()


def load_raw_train_dataset() -> pd.DataFrame:
    """Load the raw train partition created before exploratory analysis."""
    return CSVDatasetRepository(
        CSVDataSourceConfig(path=settings.raw_train_path)
    ).load()


def load_external_test_dataset() -> pd.DataFrame:
    """Load the persisted raw external test partition."""
    return CSVDatasetRepository(CSVDataSourceConfig(path=settings.raw_test_path)).load()


def current_git_commit() -> str:
    """Return the current commit when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
