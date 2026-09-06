import pandas as pd
from fastapi.testclient import TestClient

from credit_risk_lab.application import RawLoanScorer, create_deployment_split
from credit_risk_lab.config.settings import settings
from credit_risk_lab.infrastructure.modeling import load_model_bundle
from credit_risk_lab.interfaces.api import app


def raw_application() -> dict:
    """Return one valid raw row using only application-time API fields."""
    return {
        "person_age": 30,
        "person_gender": "female",
        "person_education": "Bachelor",
        "person_income": 50000,
        "person_emp_exp": 8,
        "person_home_ownership": "RENT",
        "loan_amnt": 10000,
        "loan_intent": "MEDICAL",
        "loan_int_rate": 10.0,
        "loan_percent_income": 0.2,
        "cb_person_cred_hist_length": 5,
        "credit_score": 650,
        "previous_loan_defaults_on_file": "No",
    }


def test_raw_scorer_runs_feature_engineering_before_prediction():
    result = RawLoanScorer(load_model_bundle(settings.model_bundle_path)).score(
        pd.DataFrame([raw_application()])
    )
    assert len(result.probabilities) == 1
    assert 0 <= result.probabilities[0] <= 1
    assert result.decisions[0] in {0, 1}


def test_prediction_endpoint_returns_auditable_response():
    with TestClient(app) as client:
        response = client.post("/v1/predict", json=raw_application())
    assert response.status_code == 200
    body = response.json()
    assert {
        "request_id", "model_name", "model_version", "probability_of_risk",
        "risk_decision", "risk_label", "threshold", "threshold_source",
        "scored_at_utc", "latency_ms",
    }.issubset(body)
    assert body["threshold"] == settings.decision_threshold
    assert body["threshold_source"] == "configs/settings.yaml:decision_threshold"


def test_prediction_endpoint_rejects_implausible_experience():
    payload = raw_application()
    payload["person_age"] = 20
    payload["person_emp_exp"] = 10
    with TestClient(app) as client:
        response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422


def test_deployment_split_persists_ten_percent(tmp_path):
    rows = []
    for i in range(100):
        row = raw_application()
        row["loan_status"] = i % 2
        row["person_income"] += i
        rows.append(row)
    result = create_deployment_split(
        pd.DataFrame(rows), train_path=tmp_path / "train.csv", test_path=tmp_path / "test.csv"
    )
    assert result.train_rows == 90
    assert result.test_rows == 10
    assert "loan_status" in pd.read_csv(result.test_path)
