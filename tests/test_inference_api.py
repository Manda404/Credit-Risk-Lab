import pandas as pd
from fastapi.testclient import TestClient
import pytest

from credit_risk_lab.application import (
    BatchInferenceRunner,
    InferenceInputValidator,
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
        response = client.post(
            "/v1/predict",
            json=raw_application(),
            headers={"X-Request-ID": "test-request-001"},
        )
    assert response.status_code == 200
    body = response.json()
    assert {
        "request_id",
        "model_name",
        "model_version",
        "probability_of_risk",
        "risk_decision",
        "risk_label",
        "risk_band",
        "threshold",
        "threshold_source",
        "validation_warnings",
        "scored_at_utc",
        "latency_ms",
    }.issubset(body)
    assert response.headers["X-Request-ID"] == "test-request-001"
    assert body["request_id"] == "test-request-001"
    assert body["threshold"] == settings.decision_threshold
    assert body["threshold_source"] == "configs/settings.yaml:decision_threshold"


def test_health_endpoint_exposes_runtime_environment():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["environment"] == settings.environment
    assert response.json()["version"] == settings.project_version


def test_ready_endpoint_exposes_model_artifact_metadata():
    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_bundle_path"] == str(settings.model_bundle_path)
    assert len(body["model_bundle_sha256"]) == 64


def test_prediction_endpoint_rejects_implausible_experience():
    payload = raw_application()
    payload["person_age"] = 20
    payload["person_emp_exp"] = 10
    with TestClient(app) as client:
        response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "REQUEST_VALIDATION_ERROR"
    assert "Employment experience is implausible" in str(body["details"])


def test_prediction_endpoint_returns_structured_error_for_blank_category():
    payload = raw_application()
    payload["loan_intent"] = " "

    with TestClient(app) as client:
        response = client.post("/v1/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INFERENCE_INPUT"
    assert "loan_intent" in response.json()["message"]


def test_batch_prediction_endpoint_preserves_order_and_request_id():
    payload = {"applications": [raw_application(), raw_application()]}

    with TestClient(app) as client:
        response = client.post(
            "/v1/predict/batch",
            json=payload,
            headers={"X-Request-ID": "batch-request-001"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "batch-request-001"
    assert body["rows"] == 2
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["request_id"] == "batch-request-001-000001"


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


def test_inference_input_validator_rejects_invalid_rows():
    valid = raw_application()
    invalid = raw_application()
    invalid["person_age"] = 20
    invalid["person_emp_exp"] = 10

    result = InferenceInputValidator().validate(pd.DataFrame([valid, invalid]))

    assert result.accepted_rows == 1
    assert result.rejected_count == 1
    assert "implausible" in result.rejected_rows["rejection_reason"].iloc[0]


def test_inference_input_validator_warns_on_unknown_category_without_rejecting():
    row = raw_application()
    row["loan_intent"] = "SPACE_TRAVEL"

    result = InferenceInputValidator().validate(pd.DataFrame([row]))

    assert result.accepted_rows == 1
    assert result.rejected_count == 0
    assert result.warning_count == 1
    assert "loan_intent" in result.warning_rows["warning_reason"].iloc[0]


def test_inference_input_validator_rejects_empty_category():
    row = raw_application()
    row["loan_intent"] = " "

    result = InferenceInputValidator().validate(pd.DataFrame([row]))

    assert result.accepted_rows == 0
    assert result.rejected_count == 1
    assert result.warning_count == 0
    assert "must not be empty" in result.rejected_rows["rejection_reason"].iloc[0]


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
    assert result.warning_rows.empty


def test_batch_inference_runner_skips_invalid_rows(tmp_path):
    valid = raw_application()
    invalid = raw_application()
    invalid["credit_score"] = 100

    scorer = RawLoanScorer(load_model_bundle(settings.model_bundle_path))
    runner = BatchInferenceRunner(scorer)
    result = runner.predict(
        pd.DataFrame([valid, invalid]),
        output_path=tmp_path / "submission.csv",
    )

    assert len(result.submission) == 1
    assert len(result.rejected_rows) == 1
    assert result.warning_rows.empty
    assert "credit_score" in result.rejected_rows["rejection_reason"].iloc[0]


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
