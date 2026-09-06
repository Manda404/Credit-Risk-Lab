"""Create the external holdout used to simulate production traffic."""

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.data_quality import clean_implausible_rows


@dataclass(frozen=True)
class DeploymentSplitResult:
    """Paths and row counts of the persisted 90/10 deployment split."""

    train_path: Path
    test_path: Path
    train_rows: int
    test_rows: int


def create_deployment_split(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.10,
    target_column: str = settings.target_column,
    random_state: int = settings.random_state,
    train_path: Path = settings.train_path,
    test_path: Path = settings.test_path,
) -> DeploymentSplitResult:
    """Clean, stratify, and persist a 90/10 external holdout.

    ``test.csv`` remains raw (no engineered features) so the simulator exercises
    exactly the same validation and feature-engineering code as a future client.
    The target is retained only to compare simulated predictions afterward; the
    simulator removes it from every API request.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")
    clean = clean_implausible_rows(frame)
    train, test = train_test_split(
        clean,
        test_size=test_size,
        random_state=random_state,
        stratify=clean[target_column],
    )
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return DeploymentSplitResult(train_path, test_path, len(train), len(test))
