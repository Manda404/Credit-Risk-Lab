"""
drift_analysis.py

Module for detecting dataset drift between two distributions:
typically TRAIN vs TEST or TRAIN vs PRODUCTION.

Drift metrics implemented:
--------------------------
✔ PSI (Population Stability Index)
✔ KS statistic (Kolmogorov–Smirnov)
✔ Hellinger distance

This technical module depends on pandas, numpy, and scipy and contains no
business orchestration.

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
    def _shared_edges(self, expected: np.ndarray, bins: int) -> np.ndarray:
        """Build train-derived edges with infinite tails for comparable bins."""
        finite = np.asarray(expected, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("Expected distribution has no finite values")
        edges = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)))
        if edges.size < 2:
            return np.array([-np.inf, np.inf])
        edges[0], edges[-1] = -np.inf, np.inf
        return edges

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
        cut_points = self._shared_edges(expected, bins)

        expected_counts = np.histogram(expected, bins=cut_points)[0]
        actual_counts = np.histogram(actual, bins=cut_points)[0]

        expected_perc = expected_counts / expected_counts.sum()
        actual_perc = actual_counts / actual_counts.sum()

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
    def hellinger(
        self, expected: np.ndarray, actual: np.ndarray, bins: int = None
    ) -> float:
        """
        Measures similarity between two histograms.
        Range : 0 (identical) → 1 (completely different)
        """
        bins = bins or self.bins

        edges = self._shared_edges(expected, bins)
        hist_e, _ = np.histogram(expected, bins=edges)
        hist_a, _ = np.histogram(actual, bins=edges)

        # Normalize
        hist_e = hist_e / np.sum(hist_e)
        hist_a = hist_a / np.sum(hist_a)

        # Avoid zero issues
        hist_e = np.where(hist_e == 0, 1e-8, hist_e)
        hist_a = np.where(hist_a == 0, 1e-8, hist_a)

        # Hellinger formula
        hellinger_dist = (1 / np.sqrt(2)) * np.sqrt(
            np.sum((np.sqrt(hist_e) - np.sqrt(hist_a)) ** 2)
        )
        return float(hellinger_dist)

    def categorical_psi(self, expected: pd.Series, actual: pd.Series) -> float:
        """Compute PSI on the union of categories, including missing values."""
        exp = expected.astype("string").fillna("<missing>")
        act = actual.astype("string").fillna("<missing>")
        categories = exp.unique().tolist() + [
            v for v in act.unique() if v not in set(exp.unique())
        ]
        exp_p = (
            exp.value_counts(normalize=True)
            .reindex(categories, fill_value=0)
            .to_numpy()
        )
        act_p = (
            act.value_counts(normalize=True)
            .reindex(categories, fill_value=0)
            .to_numpy()
        )
        exp_p = np.maximum(exp_p, 1e-4)
        act_p = np.maximum(act_p, 1e-4)
        return float(np.sum((exp_p - act_p) * np.log(exp_p / act_p)))

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
            features = list(expected_df.columns)
        missing = sorted(set(features).difference(actual_df.columns))
        if missing:
            raise ValueError(f"Actual dataset is missing monitored features: {missing}")

        report = {}

        for col in features:
            if not pd.api.types.is_numeric_dtype(expected_df[col]):
                report[col] = {
                    "type": "categorical",
                    "psi": self.categorical_psi(expected_df[col], actual_df[col]),
                }
                continue
            exp_vals = expected_df[col].dropna().to_numpy(dtype=float)
            act_vals = actual_df[col].dropna().to_numpy(dtype=float)

            if len(exp_vals) == 0 or len(act_vals) == 0:
                continue

            report[col] = {
                "type": "numeric",
                "psi": self.psi(exp_vals, act_vals),
                "ks": self.ks(exp_vals, act_vals),
                "hellinger": self.hellinger(exp_vals, act_vals),
            }

        return report

    def report_frame(
        self,
        expected_df: pd.DataFrame,
        actual_df: pd.DataFrame,
        *,
        features: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return a flat, sortable drift report with an interpretable PSI status.

        PSI thresholds are conventional diagnostics, not universal statistical
        laws: below 0.10 is labelled stable, 0.10–0.25 review, and above 0.25
        alert. They should be tuned to feature criticality and sample size.
        """
        nested = self.compute_drift_report(expected_df, actual_df, features=features)
        frame = (
            pd.DataFrame.from_dict(nested, orient="index")
            .rename_axis("feature")
            .reset_index()
        )
        frame["status"] = pd.cut(
            frame["psi"],
            bins=[-np.inf, 0.10, 0.25, np.inf],
            labels=["stable", "review", "alert"],
            right=False,
        ).astype(str)
        return frame.sort_values("psi", ascending=False).reset_index(drop=True)
