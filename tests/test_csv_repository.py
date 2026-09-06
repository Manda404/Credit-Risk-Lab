import pandas as pd
import pytest

from credit_risk_lab.infrastructure.data_sources import CSVDataSourceConfig, CSVDatasetRepository


def test_repository_uses_explicit_path_and_csv_options(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("id;amount\n1;10.5\n2;20.0\n", encoding="utf-8")
    config = CSVDataSourceConfig(
        path=path, sep=";", encoding="utf-8", nrows=1, dtype={"id": "string"}
    )
    repository = CSVDatasetRepository(config)
    frame = repository.load()
    assert repository.csv_path == path.resolve()
    assert len(frame) == 1
    assert str(frame["id"].dtype).startswith("string")


def test_repository_fails_for_missing_explicit_path(tmp_path):
    repository = CSVDatasetRepository(CSVDataSourceConfig(path=tmp_path / "missing.csv"))
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        repository.load()


def test_repository_saves_to_explicit_destination(tmp_path):
    destination = tmp_path / "nested" / "output.csv"
    returned = CSVDatasetRepository.save(pd.DataFrame({"value": [1]}), destination)
    assert returned == destination.resolve()
    assert pd.read_csv(destination).iloc[0, 0] == 1
