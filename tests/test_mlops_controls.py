import numpy as np
import pandas as pd
import pytest

from credit_risk_lab.application import three_way_temporal_split
from credit_risk_lab.infrastructure.analytics import DriftAnalyzer
from credit_risk_lab.infrastructure.evaluation import fairness_report
from credit_risk_lab.infrastructure.visualization import plot_drift_summary


def test_temporal_split_preserves_borrower_groups_and_order():
    frame = pd.DataFrame({
        "borrower": ["a", "a", "b", "c", "d", "e", "f", "g", "h", "i"],
        "decision_date": pd.date_range("2024-01-01", periods=10),
        "feature": range(10),
        "loan_status": [0, 1] * 5,
    })
    split = three_way_temporal_split(
        frame, date_column="decision_date", group_column="borrower"
    )
    groups = [set(part["borrower"]) for part in (split.x_train, split.x_validation, split.x_test)]
    assert groups[0].isdisjoint(groups[1] | groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert split.x_train["decision_date"].max() < split.x_test["decision_date"].min()


def test_temporal_split_rejects_invalid_dates():
    frame = pd.DataFrame({"date": ["bad", "2024-01-01"], "loan_status": [0, 1]})
    with pytest.raises(ValueError, match="invalid timestamps"):
        three_way_temporal_split(frame, date_column="date")


def test_drift_uses_shared_bins_and_detects_unseen_category():
    analyzer = DriftAnalyzer(bins=4)
    expected = pd.DataFrame({"number": np.arange(100), "category": ["a"] * 100})
    actual = pd.DataFrame({"number": np.arange(100, 200), "category": ["new"] * 100})
    report = analyzer.compute_drift_report(expected, actual)
    assert report["number"]["psi"] > 0
    assert report["number"]["hellinger"] > 0
    assert report["category"]["psi"] > 0
    frame = analyzer.report_frame(expected, actual)
    assert {"feature", "type", "psi", "status"}.issubset(frame.columns)
    assert set(frame["status"]).issubset({"stable", "review", "alert"})
    assert len(plot_drift_summary(frame).data) > 0


def test_fairness_report_aligns_rows_and_flags_small_groups():
    report = fairness_report(
        [0, 1, 0, 1], np.array([0.1, 0.9, 0.8, 0.2]),
        pd.DataFrame({"group": ["a", "a", "b", "b"]}), 0.5,
    )
    assert set(report["group"]) == {"a", "b"}
    assert report["small_group"].all()
