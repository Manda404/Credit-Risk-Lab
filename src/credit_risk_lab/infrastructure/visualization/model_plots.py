"""Reusable Plotly charts kept outside notebooks."""

import optuna
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from plotly.subplots import make_subplots


def plot_learning_curves(histories: dict[str, dict[str, list[float]]]) -> go.Figure:
    """Plot train and validation log-loss together, one panel per model."""
    names = [
        name
        for name, history in histories.items()
        if history.get("train_loss") and history.get("validation_loss")
    ]
    if not names:
        raise ValueError("No iterative model history is available")
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=names)
    for column, name in enumerate(names, start=1):
        history = histories[name]
        fig.add_trace(
            go.Scatter(
                y=history["train_loss"],
                mode="lines",
                name=f"{name} train",
                line={"color": "#1f77b4"},
            ),
            row=1,
            col=column,
        )
        fig.add_trace(
            go.Scatter(
                y=history["validation_loss"],
                mode="lines",
                name=f"{name} validation",
                line={"color": "#ff7f0e"},
            ),
            row=1,
            col=column,
        )
        fig.update_xaxes(title_text="Boosting iteration", row=1, col=column)
        fig.update_yaxes(title_text="Log loss", row=1, col=column)
    fig.update_layout(
        title="Training and validation loss",
        template="plotly_white",
        height=450,
        width=1250,
    )
    return fig


def plot_model_comparison(metrics: pd.DataFrame) -> go.Figure:
    """Compare candidate validation metrics across models."""
    selected = metrics[["model", "roc_auc", "pr_auc", "ks", "f1"]].melt(
        id_vars="model", var_name="metric", value_name="value"
    )
    return px.bar(
        selected,
        x="metric",
        y="value",
        color="model",
        barmode="group",
        range_y=[0, 1],
        title="Validation-set candidate comparison",
        template="plotly_white",
    )


def plot_feature_importance(
    importance: pd.DataFrame,
    *,
    title: str = "Feature importance",
) -> go.Figure:
    """Plot normalized model feature importances with percentage labels."""
    required = {"feature", "importance_pct"}
    missing = required.difference(importance.columns)
    if missing:
        raise ValueError(f"Feature importance DataFrame is missing: {sorted(missing)}")
    frame = importance.sort_values("importance_pct", ascending=True)
    fig = px.bar(
        frame,
        x="importance_pct",
        y="feature",
        orientation="h",
        text="importance_pct",
        title=title,
        template="plotly_white",
        color="importance_pct",
        color_continuous_scale="Blues",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        xaxis_title="Importance (%)",
        yaxis_title="Feature",
        height=max(480, 28 * len(frame) + 180),
        coloraxis_showscale=False,
        margin={"l": 180, "r": 80, "t": 80, "b": 60},
    )
    return fig


def plot_metric_improvement(
    baseline_metrics: pd.DataFrame,
    tuned_metrics: pd.DataFrame,
    *,
    model_name: str,
) -> go.Figure:
    """Compare validation metrics before and after hyperparameter tuning."""
    metrics = ["roc_auc", "pr_auc", "ks", "f1", "mcc", "cohen_kappa"]
    baseline_row = baseline_metrics.loc[baseline_metrics["model"].eq(model_name)]
    tuned_row = tuned_metrics.loc[tuned_metrics["model"].eq(model_name)]
    if baseline_row.empty:
        raise ValueError(f"Missing baseline metrics for model: {model_name}")
    if tuned_row.empty:
        raise ValueError(f"Missing tuned metrics for model: {model_name}")
    available = [
        metric
        for metric in metrics
        if metric in baseline_metrics.columns and metric in tuned_metrics.columns
    ]
    rows = []
    for metric in available:
        baseline_value = float(baseline_row.iloc[0][metric])
        tuned_value = float(tuned_row.iloc[0][metric])
        rows.extend(
            [
                {"metric": metric, "stage": "Baseline", "value": baseline_value},
                {"metric": metric, "stage": "Optuna tuned", "value": tuned_value},
            ]
        )
    frame = pd.DataFrame(rows)
    fig = px.bar(
        frame,
        x="metric",
        y="value",
        color="stage",
        barmode="group",
        text="value",
        title=f"Validation metrics: {model_name} baseline vs Optuna tuned",
        template="plotly_white",
        color_discrete_map={"Baseline": "#4E79A7", "Optuna tuned": "#F28E2B"},
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(yaxis={"range": [0, 1]}, height=480)
    return fig


def plot_optuna_trials(trials: pd.DataFrame, *, metric: str) -> go.Figure:
    """Plot Optuna objective values and running best score by trial."""
    if trials.empty:
        raise ValueError("Optuna trials DataFrame is empty")
    frame = trials.sort_values("trial").copy()
    frame["running_best"] = frame["value"].cummax()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["trial"],
            y=frame["value"],
            mode="markers",
            name="Trial score",
            marker={"color": "#4E79A7", "size": 8},
            customdata=frame.drop(columns=["trial", "value"], errors="ignore"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["trial"],
            y=frame["running_best"],
            mode="lines+markers",
            name="Running best",
            line={"color": "#F28E2B", "width": 3},
        )
    )
    fig.update_layout(
        title=f"Optuna search progress ({metric})",
        xaxis_title="Trial",
        yaxis_title=metric,
        template="plotly_white",
        height=460,
    )
    return fig


def plot_optuna_param_importance(study: optuna.Study, *, metric: str) -> go.Figure:
    """Plot Optuna hyperparameter importances for the tuned objective."""
    importances = optuna.importance.get_param_importances(study)
    if not importances:
        raise ValueError("Optuna study has no parameter importances")
    frame = pd.DataFrame(
        [
            {"parameter": parameter, "importance": importance}
            for parameter, importance in importances.items()
        ]
    ).sort_values("importance", ascending=True)
    fig = px.bar(
        frame,
        x="importance",
        y="parameter",
        orientation="h",
        text="importance",
        title=f"Optuna hyperparameter importance ({metric})",
        template="plotly_white",
        color="importance",
        color_continuous_scale="Blues",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=460, coloraxis_showscale=False)
    return fig


def plot_optuna_parameter_slices(
    trials: pd.DataFrame,
    *,
    metric: str,
    parameters: list[str] | None = None,
) -> go.Figure:
    """Plot selected hyperparameter values against Optuna objective scores."""
    if trials.empty:
        raise ValueError("Optuna trials DataFrame is empty")
    excluded = {"trial", "value", "state", "running_best"}
    available = [
        column
        for column in trials.columns
        if column not in excluded and column != metric
    ]
    selected = [
        parameter for parameter in (parameters or available) if parameter in available
    ]
    if not selected:
        raise ValueError("No Optuna parameters available for slice plots")
    rows = (len(selected) + 1) // 2
    fig = make_subplots(rows=rows, cols=2, subplot_titles=selected)
    frame = trials.sort_values("trial").copy()
    for index, parameter in enumerate(selected, start=1):
        row = (index + 1) // 2
        col = 1 if index % 2 else 2
        fig.add_trace(
            go.Scatter(
                x=(
                    frame[parameter].astype(str)
                    if frame[parameter].dtype == "object"
                    else frame[parameter]
                ),
                y=frame["value"],
                mode="markers",
                name=parameter,
                marker={
                    "color": frame["value"],
                    "colorscale": "Viridis",
                    "size": 8,
                    "showscale": index == 1,
                    "colorbar": {"title": metric},
                },
                hovertemplate=f"{parameter}=%{{x}}<br>{metric}=%{{y:.4f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text=parameter, row=row, col=col)
        fig.update_yaxes(title_text=metric, row=row, col=col)
    fig.update_layout(
        title=f"Optuna parameter slices ({metric})",
        template="plotly_white",
        height=max(460, 260 * rows),
        showlegend=False,
    )
    return fig


def plot_calibration(calibration: pd.DataFrame, model_name: str) -> go.Figure:
    """Compare predicted probability with observed positive frequency."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"color": "#777", "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=calibration["mean_probability"],
            y=calibration["observed_rate"],
            mode="lines+markers",
            name=model_name,
            marker={"size": 7 + 18 * calibration["rows"] / calibration["rows"].max()},
            customdata=calibration[["rows", "absolute_gap"]],
            hovertemplate="Mean probability=%{x:.3f}<br>Observed rate=%{y:.3f}<br>Rows=%{customdata[0]}<br>Gap=%{customdata[1]:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Calibration curve — {model_name}",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed positive rate",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        template="plotly_white",
        width=700,
        height=550,
    )
    return fig


def plot_confusion_matrix_and_roc(
    y_true,
    probabilities,
    *,
    threshold: float,
    model_name: str,
) -> go.Figure:
    """Plot confusion matrix and ROC curve in one figure."""
    target = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    annotated_matrix = [
        [f"TN<br>{tn}", f"FP<br>{fp}"],
        [f"FN<br>{fn}", f"TP<br>{tp}"],
    ]
    fpr, tpr, _ = roc_curve(target, probabilities)
    roc_auc = roc_auc_score(target, probabilities)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Confusion matrix threshold={threshold:.3f}",
            f"ROC curve AUC={roc_auc:.3f}",
        ],
    )
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=[
                "Pred 0<br>Low risk",
                "Pred 1<br>High risk",
            ],
            y=[
                "Actual 0<br>Low risk",
                "Actual 1<br>High risk",
            ],
            colorscale="Blues",
            text=annotated_matrix,
            texttemplate="%{text}",
            showscale=False,
            hovertemplate=(
                "%{y}<br>%{x}<br>Rows=%{z}" "<extra>Confusion matrix</extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC AUC={roc_auc:.3f}",
            line={"color": "#4E79A7", "width": 3},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line={"color": "#777", "dash": "dash"},
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Predicted class", row=1, col=1)
    fig.update_yaxes(title_text="Actual class", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="False positive rate", row=1, col=2)
    fig.update_yaxes(title_text="True positive rate", row=1, col=2)
    fig.update_layout(
        title=f"Final decision diagnostics - {model_name}",
        template="plotly_white",
        height=520,
        width=1100,
    )
    return fig


def plot_lift_gain_accumulation(lift_gain: pd.DataFrame) -> go.Figure:
    """Plot cumulative gain, cumulative lift, and decile lift together."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Cumulative gain", "Cumulative lift", "Decile lift"],
    )
    fig.add_trace(
        go.Scatter(
            x=lift_gain["sample_fraction"],
            y=lift_gain["cumulative_capture_rate"],
            mode="lines+markers",
            name="Model",
            line={"color": "#4E79A7", "width": 3},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line={"color": "#777", "dash": "dash"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=lift_gain["decile"].astype(str),
            y=lift_gain["cumulative_lift"],
            mode="lines+markers",
            name="Cumulative lift",
            line={"color": "#F28E2B", "width": 3},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=lift_gain["decile"].astype(str),
            y=lift_gain["lift"],
            name="Decile lift",
            marker={"color": "#59A14F"},
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(title_text="Sample fraction", row=1, col=1)
    fig.update_yaxes(title_text="Captured positives", row=1, col=1)
    fig.update_xaxes(title_text="Decile", row=1, col=2)
    fig.update_yaxes(title_text="Lift", row=1, col=2)
    fig.update_xaxes(title_text="Decile", row=1, col=3)
    fig.update_yaxes(title_text="Lift", row=1, col=3)
    fig.update_layout(
        title="Lift, gain, and accumulation diagnostics",
        template="plotly_white",
        height=500,
        width=1250,
        showlegend=True,
    )
    return fig


def plot_metric_confidence_intervals(intervals: pd.DataFrame) -> go.Figure:
    """Plot bootstrap confidence intervals for key metrics."""
    if intervals.empty:
        raise ValueError("Metric intervals DataFrame is empty")
    frame = intervals.sort_values("estimate", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["estimate"],
            y=frame["metric"],
            mode="markers",
            marker={"color": "#4E79A7", "size": 10},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": frame["upper"] - frame["estimate"],
                "arrayminus": frame["estimate"] - frame["lower"],
            },
            name="Metric estimate",
        )
    )
    fig.update_layout(
        title="Bootstrap confidence intervals for final metrics",
        xaxis_title="Metric value",
        yaxis_title="Metric",
        xaxis={"range": [0, 1]},
        template="plotly_white",
        height=460,
    )
    return fig


def plot_threshold_tradeoff(
    threshold_grid: pd.DataFrame,
    *,
    selected_threshold: float,
) -> go.Figure:
    """Plot operational threshold trade-offs for credit-risk decisions."""
    required = {
        "threshold",
        "precision",
        "recall",
        "alert_rate",
        "approval_rate",
        "high_risk_missed",
        "false_alerts",
    }
    missing = required.difference(threshold_grid.columns)
    if missing:
        raise ValueError(f"Threshold grid is missing columns: {sorted(missing)}")
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Precision / recall",
            "Operational volume",
            "Decision errors",
        ],
    )
    fig.add_trace(
        go.Scatter(
            x=threshold_grid["threshold"],
            y=threshold_grid["recall"],
            mode="lines+markers",
            name="Recall high risk caught",
            line={"color": "#C44E52", "width": 3},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=threshold_grid["threshold"],
            y=threshold_grid["precision"],
            mode="lines+markers",
            name="Precision real alerts",
            line={"color": "#4C72B0", "width": 3},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=threshold_grid["threshold"],
            y=threshold_grid["alert_rate"],
            mode="lines+markers",
            name="Alert rate",
            line={"color": "#F28E2B", "width": 3},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=threshold_grid["threshold"],
            y=threshold_grid["approval_rate"],
            mode="lines+markers",
            name="Approval rate",
            line={"color": "#59A14F", "width": 3},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=threshold_grid["threshold"],
            y=threshold_grid["high_risk_missed"],
            name="High risk missed",
            marker={"color": "#C44E52"},
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Bar(
            x=threshold_grid["threshold"],
            y=threshold_grid["false_alerts"],
            name="False alerts",
            marker={"color": "#4C72B0"},
        ),
        row=1,
        col=3,
    )
    for column in range(1, 4):
        fig.add_vline(
            x=selected_threshold,
            line_dash="dash",
            line_color="black",
            annotation_text=f"selected={selected_threshold:.3f}",
            row=1,
            col=column,
        )
    fig.update_xaxes(title_text="Decision threshold")
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Rate", row=1, col=2)
    fig.update_yaxes(title_text="Rows", row=1, col=3)
    fig.update_layout(
        title="Credit-risk threshold trade-off",
        template="plotly_white",
        height=520,
        width=1300,
        barmode="group",
    )
    return fig
