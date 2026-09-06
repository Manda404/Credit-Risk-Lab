import pandas as pd
import pytest

from credit_risk_lab.infrastructure.data_quality import (
    build_quality_report,
    clean_implausible_rows,
)


def sample_frame():
    return pd.DataFrame(
        {
            "person_age": [30, 144],
            "person_gender": ["female", "male"],
            "person_education": ["Bachelor", "Master"],
            "person_income": [50000, 60000],
            "person_emp_exp": [8, 125],
            "person_home_ownership": ["RENT", "OWN"],
            "loan_amnt": [10000, 12000],
            "loan_intent": ["MEDICAL", "PERSONAL"],
            "loan_int_rate": [10.0, 12.0],
            "loan_percent_income": [0.2, 0.2],
            "cb_person_cred_hist_length": [5, 10],
            "credit_score": [650, 700],
            "previous_loan_defaults_on_file": ["No", "Yes"],
            "loan_status": [0, 1],
        }
    )


def test_quality_report_flags_implausible_records():
    report = build_quality_report(sample_frame())
    assert report.invalid_age_rows == 1
    assert report.invalid_experience_rows == 1


def test_cleaner_keeps_only_plausible_records():
    assert len(clean_implausible_rows(sample_frame())) == 1


def test_quality_rejects_missing_required_column(credit_risk_sample):
    frame = credit_risk_sample.drop(columns=["credit_score"])
    with pytest.raises(ValueError, match="credit_score"):
        build_quality_report(frame)


def test_quality_rejects_non_binary_target(credit_risk_sample):
    frame = credit_risk_sample.copy()
    frame.loc[0, "loan_status"] = 2
    with pytest.raises(ValueError, match="binary"):
        build_quality_report(frame)


def test_quality_fixture_keeps_only_plausible_invalid_rows(invalid_credit_risk_rows):
    report = build_quality_report(invalid_credit_risk_rows)
    assert report.invalid_age_rows == 1
    assert report.invalid_experience_rows == 1
    assert clean_implausible_rows(invalid_credit_risk_rows).empty
