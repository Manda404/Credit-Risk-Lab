"""
drift_analysis.py

Module for detecting dataset drift between two distributions:
typically TRAIN vs TEST or TRAIN vs PRODUCTION.

Drift metrics implemented:
--------------------------
✔ PSI (Population Stability Index)
✔ KS statistic (Kolmogorov–Smirnov)
✔ Hellinger distance

This module belongs to the Infrastructure layer because:
- depends on pandas, numpy, scipy
- purely technical (no business rules)
- does not belong to Domain or Application

Usage:
------
from credit_risk_lab.infrastructure.analytics.drift_analysis import DriftAnalyzer

analyzer = DriftAnalyzer()
report = analyzer.compute_drift_report(train_df, prod_df)

print(report["psi"])
print(report["ks"])
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, Any


class DriftAnalyzer:
    """
    DriftAnalyzer
    -------------
    Provides multiple drift detection metrics between
    two distributions (train vs test, train vs production).

    Methods available:
    - psi(): Population Stability Index
    - ks(): Kolmogorov–Smirnov statistic
    - hellinger(): Hellinger distance
    - compute_drift_report(): aggregates all metrics per feature
    """

    def __init__(self, bins: int = 10):
        self.bins = bins

    # ---------------------------------------------------------
    # 1. POPULATION STABILITY INDEX (PSI)
    # ---------------------------------------------------------
    def psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = None) -> float:
        """
        Compute PSI between two numerical distributions.

        PSI formula:
        ------------
        PSI = Σ (p_i - q_i) * ln(p_i / q_i)

        where:
        p_i = proportion of expected (train)
        q_i = proportion of actual (test/prod)
        """
        bins = bins or self.bins

        # Handle categorical or skewed distributions by quantile binning
        quantiles = np.linspace(0, 1, bins + 1)
        cut_points = np.quantile(expected, quantiles)

        # To avoid duplicate bin edges
        cut_points = np.unique(cut_points)
        if len(cut_points) <= 2:
            return 0.0

        expected_counts = np.histogram(expected, bins=cut_points)[0]
        actual_counts = np.histogram(actual, bins=cut_points)[0]

        expected_perc = expected_counts / len(expected)
        actual_perc = actual_counts / len(actual)

        # avoid division by zero
        actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc)
        expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc)

        psi_value = np.sum(
            (expected_perc - actual_perc) * np.log(expected_perc / actual_perc)
        )
        return float(psi_value)

    # ---------------------------------------------------------
    # 2. KS STATISTIC (Kolmogorov–Smirnov)
    # ---------------------------------------------------------
    def ks(self, expected: np.ndarray, actual: np.ndarray) -> float:
        """
        KS drift metric.
        Higher KS → stronger drift.

        Values:
        -------
        0.00–0.10 → No drift
        0.10–0.20 → Moderate drift
        > 0.20   → Strong drift
        """
        statistic, _ = ks_2samp(expected, actual)
        return float(statistic)

    # ---------------------------------------------------------
    # 3. HELLINGER DISTANCE
    # ---------------------------------------------------------
    def hellinger(self, expected: np.ndarray, actual: np.ndarray, bins: int = None) -> float:
        """
        Measures similarity between two histograms.
        Range : 0 (identical) → 1 (completely different)
        """
        bins = bins or self.bins

        hist_e, _ = np.histogram(expected, bins=bins, density=True)
        hist_a, _ = np.histogram(actual, bins=bins, density=True)

        # Normalize
        hist_e = hist_e / np.sum(hist_e)
        hist_a = hist_a / np.sum(hist_a)

        # Avoid zero issues
        hist_e = np.where(hist_e == 0, 1e-8, hist_e)
        hist_a = np.where(hist_a == 0, 1e-8, hist_a)

        # Hellinger formula
        hellinger_dist = (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(hist_e) - np.sqrt(hist_a)) ** 2))
        return float(hellinger_dist)

    # ---------------------------------------------------------
    # 4. FULL REPORT
    # ---------------------------------------------------------
    def compute_drift_report(
        self,
        expected_df: pd.DataFrame,
        actual_df: pd.DataFrame,
        *,
        features: list[str] | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute PSI, KS, and Hellinger distance for ALL numerical features.

        Returns:
        --------
        {
          "age": {"psi": 0.04, "ks": 0.12, "hellinger": 0.08},
          "income": {"psi": 0.21, "ks": 0.32, "hellinger": 0.11},
        }
        """
        if features is None:
            # automatically detect numeric columns
            features = [
                col
                for col in expected_df.columns
                if pd.api.types.is_numeric_dtype(expected_df[col])
            ]

        report = {}

        for col in features:
            exp_vals = expected_df[col].dropna().to_numpy()
            act_vals = actual_df[col].dropna().to_numpy()

            if len(exp_vals) == 0 or len(act_vals) == 0:
                continue

            report[col] = {
                "psi": self.psi(exp_vals, act_vals),
                "ks": self.ks(exp_vals, act_vals),
                "hellinger": self.hellinger(exp_vals, act_vals),
            }

        return report
