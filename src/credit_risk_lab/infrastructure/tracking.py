"""MLflow experiment tracking with explicit dataset and governance metadata."""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import mlflow

from credit_risk_lab.config.settings import settings


@contextmanager
def tracked_run(
    *,
    run_name: str,
    parameters: dict,
    experiment_name: str | None = None,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Create an MLflow run and record parameters/tags consistently.

    A local ``mlruns`` directory is suitable for teaching only. A governed
    deployment should pass a shared tracking URI with access control, retention,
    registry permissions, and backup.
    """
    mlflow.set_tracking_uri(tracking_uri or settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name or settings.mlflow_experiment_name)
    resolved_tags = {"environment": settings.environment, **(tags or {})}
    with mlflow.start_run(run_name=run_name, tags=resolved_tags) as run:
        mlflow.log_params({k: str(v) for k, v in parameters.items()})
        yield run


def log_training_outputs(
    metrics: dict[str, float], artifacts: list[Path] | None = None
) -> None:
    """Log numeric test metrics and existing report files to the active run."""
    mlflow.log_metrics(
        {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    )
    for artifact in artifacts or []:
        if artifact.exists():
            mlflow.log_artifact(str(artifact))
