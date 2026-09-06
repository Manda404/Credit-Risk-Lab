"""Reusable Plotly charts kept outside notebooks."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_learning_curves(histories: dict[str, dict[str, list[float]]]) -> go.Figure:
    """Plot train and validation log-loss together, one panel per model."""
    names = [
        name for name, history in histories.items()
        if history.get("train_loss") and history.get("validation_loss")
    ]
    if not names:
        raise ValueError("No iterative model history is available")
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=names)
    for column, name in enumerate(names, start=1):
        history = histories[name]
        fig.add_trace(go.Scatter(y=history["train_loss"], mode="lines", name=f"{name} train", line={"color": "#1f77b4"}), row=1, col=column)
        fig.add_trace(go.Scatter(y=history["validation_loss"], mode="lines", name=f"{name} validation", line={"color": "#ff7f0e"}), row=1, col=column)
        fig.update_xaxes(title_text="Boosting iteration", row=1, col=column)
        fig.update_yaxes(title_text="Log loss", row=1, col=column)
    fig.update_layout(title="Training and validation loss", template="plotly_white", height=450, width=1250)
    return fig


def plot_model_comparison(metrics: pd.DataFrame) -> go.Figure:
    """Compare candidate validation metrics across models."""
    selected = metrics[["model", "roc_auc", "pr_auc", "ks", "f1"]].melt(id_vars="model", var_name="metric", value_name="value")
    return px.bar(selected, x="metric", y="value", color="model", barmode="group", range_y=[0, 1], title="Validation-set candidate comparison", template="plotly_white")


def plot_calibration(calibration: pd.DataFrame, model_name: str) -> go.Figure:
    """Compare predicted probability with observed positive frequency."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
        line={"color": "#777", "dash": "dash"},
    ))
    fig.add_trace(go.Scatter(
        x=calibration["mean_probability"], y=calibration["observed_rate"],
        mode="lines+markers", name=model_name,
        marker={"size": 7 + 18 * calibration["rows"] / calibration["rows"].max()},
        customdata=calibration[["rows", "absolute_gap"]],
        hovertemplate="Mean probability=%{x:.3f}<br>Observed rate=%{y:.3f}<br>Rows=%{customdata[0]}<br>Gap=%{customdata[1]:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Calibration curve — {model_name}", xaxis_title="Mean predicted probability",
        yaxis_title="Observed positive rate", xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]}, template="plotly_white", width=700, height=550,
    )
    return fig
