import json
from pathlib import Path

import pandas as pd
import pytest


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def credit_risk_sample() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "credit_risk_sample.csv")


@pytest.fixture
def invalid_credit_risk_rows() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "credit_risk_invalid_rows.csv")


@pytest.fixture
def expected_schema() -> dict:
    return json.loads((FIXTURES / "expected_schema.json").read_text(encoding="utf-8"))


@pytest.fixture
def expected_features() -> dict:
    return json.loads((FIXTURES / "expected_features.json").read_text(encoding="utf-8"))
