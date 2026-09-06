"""MLflow experiment tracking with explicit dataset and governance metadata."""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import mlflow


@contextmanager
def tracked_run(
    *,
    experiment_name: str,
    run_name: str,
    parameters: dict,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Create an MLflow run and record parameters/tags consistently.

    A local ``mlruns`` directory is suitable for teaching only. A governed
    deployment should pass a shared tracking URI with access control, retention,
    registry permissions, and backup.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        mlflow.log_params({k: str(v) for k, v in parameters.items()})
        yield run


def log_training_outputs(metrics: dict[str, float], artifacts: list[Path] | None = None) -> None:
    """Log numeric test metrics and existing report files to the active run."""
    mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
    for artifact in artifacts or []:
        if artifact.exists():
            mlflow.log_artifact(str(artifact))
