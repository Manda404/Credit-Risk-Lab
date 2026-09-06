"""Compact Plotly visuals for train/test drift diagnostics."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DRIFT_COLORS = {"stable": "#2ca02c", "review": "#ffbf00", "alert": "#d62728"}


def plot_drift_summary(report: pd.DataFrame) -> go.Figure:
    """Plot PSI by feature with conventional review and alert thresholds."""
    ordered = report.sort_values("psi", ascending=True)
    figure = px.bar(
        ordered, x="psi", y="feature", orientation="h", color="status",
        color_discrete_map=DRIFT_COLORS, hover_data=["type", "ks", "hellinger"],
        title="Train vs external test drift — PSI by feature", template="plotly_white",
    )
    figure.add_vline(x=0.10, line_dash="dash", line_color="#ffbf00", annotation_text="review 0.10")
    figure.add_vline(x=0.25, line_dash="dash", line_color="#d62728", annotation_text="alert 0.25")
    figure.update_layout(height=max(500, 28 * len(report)), xaxis_title="Population Stability Index")
    return figure


def plot_numeric_distribution(train: pd.DataFrame, test: pd.DataFrame, feature: str) -> go.Figure:
    """Overlay normalized histograms for one numerical feature."""
    frame = pd.concat([
        pd.DataFrame({feature: train[feature], "sample": "train"}),
        pd.DataFrame({feature: test[feature], "sample": "external_test"}),
    ], ignore_index=True)
    return px.histogram(
        frame, x=feature, color="sample", histnorm="probability density",
        barmode="overlay", opacity=0.55, nbins=40,
        title=f"Distribution comparison — {feature}", template="plotly_white",
    )


def plot_categorical_distribution(train: pd.DataFrame, test: pd.DataFrame, feature: str) -> go.Figure:
    """Compare normalized category frequencies for one categorical feature."""
    rows = []
    for sample, series in (("train", train[feature]), ("external_test", test[feature])):
        frequency = series.astype("string").fillna("<missing>").value_counts(normalize=True)
        rows.extend({"category": str(k), "frequency": float(v), "sample": sample} for k, v in frequency.items())
    return px.bar(
        pd.DataFrame(rows), x="category", y="frequency", color="sample",
        barmode="group", title=f"Category comparison — {feature}", template="plotly_white",
    )
