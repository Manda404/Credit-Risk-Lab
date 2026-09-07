import numpy as np
import pandas as pd

from credit_risk_lab.infrastructure.evaluation import (
    ThresholdAnalyzer,
    bootstrap_metric_intervals,
    calibration_table,
    classification_metrics,
    expected_calibration_error,
    find_optimal_threshold,
    find_cost_sensitive_threshold,
    lift_gain_table,
)
from credit_risk_lab.infrastructure.modeling import build_preprocessor


def test_preprocessor_handles_unknown_categories():
    train = pd.DataFrame({"number": [1.0, 2.0], "category": ["a", "b"]})
    test = pd.DataFrame({"number": [3.0], "category": ["unknown"]})
    transformer = build_preprocessor(train)
    assert transformer.fit_transform(train).shape[0] == 2
    assert transformer.transform(test).shape[0] == 1


def test_metrics_and_threshold_are_valid():
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])
    threshold = find_optimal_threshold(target, probabilities)
    metrics = classification_metrics(target, probabilities, threshold)
    assert 0 <= threshold <= 1
    assert metrics["roc_auc"] == 1.0
    assert {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "log_loss",
        "brier",
        "mcc",
        "cohen_kappa",
        "ece",
    }.issubset(metrics)
    costly_miss_threshold = find_cost_sensitive_threshold(
        target, probabilities, false_positive_cost=1, false_negative_cost=10
    )
    assert 0 <= costly_miss_threshold <= 1


def test_ece_is_zero_for_perfectly_calibrated_groups():
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])
    table = calibration_table(target, probabilities, bins=10)
    assert table["rows"].sum() == 4
    assert expected_calibration_error(target, probabilities, bins=10) == 0.0


def test_lift_gain_and_bootstrap_intervals_are_valid():
    target = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8, 0.35, 0.7, 0.4, 0.6])
    lift_gain = lift_gain_table(target, probabilities, bins=4)
    intervals = bootstrap_metric_intervals(
        target,
        probabilities,
        threshold=0.5,
        n_bootstrap=20,
        random_state=7,
    )

    assert lift_gain["cumulative_capture_rate"].iloc[-1] == 1.0
    assert {"lift", "cumulative_lift", "sample_fraction"}.issubset(lift_gain.columns)
    assert {"metric", "estimate", "lower", "upper"}.issubset(intervals.columns)


def test_threshold_analyzer_returns_credit_risk_tradeoff_grid():
    target = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.3, 0.6])
    grid = ThresholdAnalyzer().grid(target, probabilities)

    assert {"high_risk_caught", "high_risk_missed", "false_alerts"}.issubset(
        grid.columns
    )
    assert grid["threshold"].min() == 0.02
    assert grid["precision"].between(0, 1).all()
    assert grid["recall"].between(0, 1).all()
