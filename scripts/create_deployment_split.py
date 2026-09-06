"""Create raw train.csv/test.csv where test is a production-simulation holdout."""

from credit_risk_lab.application.workflows import run_split_and_drift_workflow
from credit_risk_lab.config.settings import settings


def main() -> None:
    """Run the packaged split workflow and print its locations."""
    result = run_split_and_drift_workflow()
    print(f"train={settings.train_path} rows={len(result.train)}")
    print(f"test={settings.test_path} rows={len(result.external_test)}")


if __name__ == "__main__":
    main()
