import pandas as pd

from credit_risk_lab.infrastructure.feature_engineering import LoanFeatureEngineer


def sample_frame():
    return pd.DataFrame(
        {
            "person_age": [30],
            "person_gender": ["female"],
            "person_education": ["Bachelor"],
            "person_income": [50000],
            "person_emp_exp": [8],
            "person_home_ownership": ["RENT"],
            "loan_amnt": [10000],
            "loan_intent": ["MEDICAL"],
            "loan_int_rate": [10.0],
            "loan_percent_income": [0.2],
            "cb_person_cred_hist_length": [5],
            "credit_score": [650],
            "previous_loan_defaults_on_file": ["No"],
        }
    )


def test_feature_engineer_adds_expected_features():
    transformed = LoanFeatureEngineer().transform(sample_frame())
    assert {"dti", "log_income", "credit_score_band", "has_default_before"}.issubset(
        transformed.columns
    )
    assert len(transformed) == 1
    # norm_credit_score is intentionally not produced here: computing it from
    # whichever frame is passed in would leak train/validation/test statistics
    # into a feature. Scaling is instead handled by the leakage-safe preprocessor.
    assert "norm_credit_score" not in transformed.columns


def test_new_domain_features_are_computed_correctly():
    transformed = LoanFeatureEngineer().transform(sample_frame())
    row = transformed.iloc[0]

    assert {
        "age_first_credit",
        "emp_credit_hist_gap",
        "rate_per_score_point",
        "income_per_experience_year",
        "risk_flags_count",
    }.issubset(transformed.columns)

    assert row["age_first_credit"] == 30 - 5
    assert row["emp_credit_hist_gap"] == 8 - 5
    assert row["rate_per_score_point"] == 10.0 / 650
    assert row["income_per_experience_year"] == 50000 / (8 + 1)
    # Only the RENT flag is triggered: no previous default, score >= 580,
    # loan_percent_income <= 0.40, and intent is not DEBTCONSOLIDATION.
    assert row["risk_flags_count"] == 1


def test_risk_flags_count_accumulates_independent_conditions():
    frame = sample_frame()
    frame["credit_score"] = 500
    frame["loan_percent_income"] = 0.5
    frame["previous_loan_defaults_on_file"] = "Yes"
    frame["loan_intent"] = "DEBTCONSOLIDATION"
    transformed = LoanFeatureEngineer().transform(frame)
    assert transformed.iloc[0]["risk_flags_count"] == 5


def test_feature_engineering_matches_versioned_feature_contract(
    credit_risk_sample, expected_features
):
    transformed = LoanFeatureEngineer().transform(
        credit_risk_sample.drop(columns=["loan_status"])
    )
    for feature in expected_features["engineered_features"]:
        assert feature in transformed.columns
    assert transformed[expected_features["engineered_features"]].notna().all().all()
