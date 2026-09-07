"""Create train/validation artifacts from the raw train partition."""

from credit_risk_lab.application.workflows import run_split_and_drift_workflow
from credit_risk_lab.config.settings import settings


def main() -> None:
    """Run the packaged split workflow and print its locations."""
    result = run_split_and_drift_workflow()
    print(f"train={settings.train_path} rows={len(result.train)}")
    print(f"validation={settings.validation_path} rows={len(result.validation)}")


if __name__ == "__main__":
    main()
