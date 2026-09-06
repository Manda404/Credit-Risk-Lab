"""Create raw train.csv/test.csv where test is a 10% production simulation holdout."""

from credit_risk_lab.application import create_deployment_split
from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.data_sources import CSVDataSourceConfig, CSVDatasetRepository


def main() -> None:
    """Load the configured raw CSV, create the split, and print its locations."""
    source = CSVDataSourceConfig(
        path=settings.raw_data_path,
        sep=settings.raw_data_sep,
        encoding=settings.raw_data_encoding,
    )
    result = create_deployment_split(CSVDatasetRepository(source).load(), test_size=0.10)
    print(f"train={result.train_path} rows={result.train_rows}")
    print(f"test={result.test_path} rows={result.test_rows}")


if __name__ == "__main__":
    main()
