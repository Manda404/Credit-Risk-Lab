"""Build the holdout, drift report, training run, and model bundle."""

from credit_risk_lab.application.workflows import (
    run_split_and_drift_workflow,
    run_training_workflow,
)


def main() -> None:
    """Execute deployment preparation through application workflows."""
    drift = run_split_and_drift_workflow()
    training = run_training_workflow()
    print(
        "Drift: "
        f"stable={(drift.drift_report['status'] == 'stable').sum()}, "
        f"review={(drift.drift_report['status'] == 'review').sum()}, "
        f"alert={(drift.drift_report['status'] == 'alert').sum()}"
    )
    print(
        "Prepared "
        f"train={len(drift.train)}, "
        f"external_test={len(drift.external_test)}, "
        f"model={training.selected_model_name}"
    )


if __name__ == "__main__":
    main()
