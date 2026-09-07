import pandas as pd

from credit_risk_lab.infrastructure.analytics import DataLeakageAuditor


def test_row_overlap_report_detects_exact_duplicate_rows():
    train = pd.DataFrame({"feature": [1, 2], "loan_status": [0, 1]})
    holdout = pd.DataFrame({"feature": [2, 3], "loan_status": [1, 0]})

    report = DataLeakageAuditor("loan_status").row_overlap_report(train, holdout)

    assert report.loc[0, "overlap_count"] == 1
    assert not bool(report.loc[0, "passed"])


def test_target_leakage_report_flags_target_column():
    features = pd.DataFrame({"feature": [1], "loan_status": [0]})

    report = DataLeakageAuditor("loan_status").target_leakage_report(features)

    assert report.loc[0, "target_present"]
    assert not bool(report.loc[0, "passed"])
