import pandas as pd
from fastapi.testclient import TestClient
import pytest

from credit_risk_lab.application import (
    BatchInferenceRunner,
    RawLoanScorer,
    RealtimeInferenceSimulator,
    create_deployment_split,
)
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
        "request_id",
        "model_name",
        "model_version",
        "probability_of_risk",
        "risk_decision",
        "risk_label",
        "threshold",
        "threshold_source",
        "scored_at_utc",
        "latency_ms",
    }.issubset(body)
    assert body["threshold"] == settings.decision_threshold
    assert body["threshold_source"] == "configs/settings.yaml:decision_threshold"


def test_health_endpoint_exposes_runtime_environment():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["environment"] == settings.environment


def test_prediction_endpoint_rejects_implausible_experience():
    payload = raw_application()
    payload["person_age"] = 20
    payload["person_emp_exp"] = 10
    with TestClient(app) as client:
        response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422


def test_deployment_split_returns_ten_percent_without_persisting(tmp_path):
    rows = []
    for i in range(100):
        row = raw_application()
        row["loan_status"] = i % 2
        row["person_income"] += i
        rows.append(row)
    result = create_deployment_split(
        pd.DataFrame(rows),
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test.csv",
    )
    assert result.train_rows == 90
    assert result.test_rows == 10
    assert "loan_status" in result.test
    assert not result.train_path.exists()
    assert not result.test_path.exists()


def test_raw_scorer_rejects_missing_inference_feature():
    frame = pd.DataFrame([raw_application()]).drop(columns=["credit_score"])
    with pytest.raises(ValueError, match="credit_score"):
        RawLoanScorer(load_model_bundle(settings.model_bundle_path)).score(frame)


def test_batch_inference_runner_writes_submission(tmp_path):
    rows = []
    for i in range(5):
        row = raw_application()
        row["loan_status"] = i % 2
        row["person_income"] += i * 100
        rows.append(row)

    scorer = RawLoanScorer(load_model_bundle(settings.model_bundle_path))
    runner = BatchInferenceRunner(scorer)
    result = runner.predict(
        pd.DataFrame(rows),
        output_path=tmp_path / "submission.csv",
        limit=3,
    )

    assert result.output_path.exists()
    assert len(result.submission) == 3
    assert {
        "request_id",
        "probability_of_risk",
        "risk_decision",
        "risk_label",
        "risk_band",
        "actual_label",
        "is_correct",
    }.issubset(result.submission.columns)


def test_realtime_inference_simulator_streams_without_real_sleep():
    pauses = []
    scorer = RawLoanScorer(load_model_bundle(settings.model_bundle_path))
    simulator = RealtimeInferenceSimulator(
        scorer,
        min_pause_seconds=1,
        max_pause_seconds=1,
        sleep_fn=pauses.append,
    )

    events = simulator.stream(
        pd.DataFrame([raw_application(), raw_application(), raw_application()]),
        limit=3,
        print_events=False,
    )

    assert len(events) == 3
    assert pauses == [1, 1]
    assert events["probability_of_risk"].between(0, 1).all()
    assert set(events["risk_decision"]).issubset({0, 1})
