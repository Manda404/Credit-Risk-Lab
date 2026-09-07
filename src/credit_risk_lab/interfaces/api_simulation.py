"""In-process API simulation helper for notebooks and demos."""

from dataclasses import dataclass

import pandas as pd
from fastapi.testclient import TestClient

from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.data_sources import (
    CSVDataSourceConfig,
    CSVDatasetRepository,
)
from credit_risk_lab.interfaces.api import app


@dataclass(frozen=True)
class ApiSimulationResult:
    """HTTP-level simulation results."""

    responses: pd.DataFrame
    invalid_status_code: int
    health: dict


def run_api_simulation(limit: int = 10) -> ApiSimulationResult:
    """Exercise the FastAPI app through its public HTTP contract."""
    test_raw = (
        CSVDatasetRepository(CSVDataSourceConfig(path=settings.raw_test_path))
        .load()
        .head(limit)
    )
    responses = []
    with TestClient(app) as client:
        health = client.get("/health").json()
        for _, row in test_raw.iterrows():
            payload = row.drop(labels=[settings.target_column]).to_dict()
            response = client.post("/v1/predict", json=payload)
            responses.append({"http": response.status_code, **response.json()})
        invalid = test_raw.iloc[0].drop(labels=[settings.target_column]).to_dict()
        invalid["person_emp_exp"] = 99
        invalid_response = client.post("/v1/predict", json=invalid)
    return ApiSimulationResult(
        responses=pd.DataFrame(responses),
        invalid_status_code=invalid_response.status_code,
        health=health,
    )
