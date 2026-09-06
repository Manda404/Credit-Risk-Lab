"""Build the 90/10 holdout and train a bundle without seeing external test rows."""

import subprocess
from credit_risk_lab.application import TrainBoostingModelsUseCase, create_deployment_split
from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.data_sources import CSVDataSourceConfig, CSVDatasetRepository
from credit_risk_lab.infrastructure.feature_engineering import LoanFeatureEngineer
from credit_risk_lab.infrastructure.modeling import save_model_bundle, sha256_file
from credit_risk_lab.infrastructure.analytics import DriftAnalyzer
from credit_risk_lab.infrastructure.visualization import plot_drift_summary


def git_commit() -> str:
    """Return the current commit when available, otherwise an explicit marker."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    """Create external holdout, train on its 90%, and persist the winner."""
    raw_source = CSVDataSourceConfig(
        path=settings.raw_data_path,
        sep=settings.raw_data_sep,
        encoding=settings.raw_data_encoding,
    )
    raw = CSVDatasetRepository(raw_source).load()
    split = create_deployment_split(raw, test_size=0.10)
    train_raw = CSVDatasetRepository(CSVDataSourceConfig(path=settings.train_path)).load()
    test_raw = CSVDatasetRepository(CSVDataSourceConfig(path=settings.test_path)).load()
    monitored_features = [column for column in train_raw.columns if column != settings.target_column]
    drift_report = DriftAnalyzer(bins=10).report_frame(
        train_raw, test_raw, features=monitored_features
    )
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    drift_report.to_csv(settings.drift_report_path, index=False)
    plot_drift_summary(drift_report).write_html(
        settings.drift_summary_plot_path, include_plotlyjs="cdn"
    )
    print(
        "Drift: "
        f"stable={(drift_report['status'] == 'stable').sum()}, "
        f"review={(drift_report['status'] == 'review').sum()}, "
        f"alert={(drift_report['status'] == 'alert').sum()}"
    )
    modeling_train = LoanFeatureEngineer().transform(train_raw)
    result = TrainBoostingModelsUseCase().execute(modeling_train)
    winner = result.selected_model_name
    test_row = result.test_metrics.iloc[0]
    save_model_bundle(
        settings.model_bundle_path,
        model=result.models[winner],
        preprocessor=result.preprocessor,
        threshold=float(test_row["threshold"]),
        metadata={
            "model_name": winner,
            "test_metrics": test_row.to_dict(),
            "selection_metric": settings.selection_metric,
            "split_strategy": result.split_strategy,
            "training_dataset_sha256": sha256_file(settings.train_path),
            "external_test_dataset_sha256": sha256_file(settings.test_path),
            "models_config_sha256": sha256_file(settings.models_config_path),
            "git_commit": git_commit(),
            "target_definition": "loan_status=1 is the synthetic positive risk class",
        },
    )
    print(f"Prepared train={split.train_rows}, external_test={split.test_rows}, model={winner}")


if __name__ == "__main__":
    main()
