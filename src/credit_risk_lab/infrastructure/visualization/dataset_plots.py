"""Dataset-level visualizations for notebook exploration."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_TARGET_LABELS = {
    0: "0 = Low risk",
    1: "1 = High risk",
}
BACKGROUND_COLOR = "#F8FAFC"
GRID_COLOR = "rgba(100,116,139,0.22)"
NEUTRAL_COLOR = "#64748B"
TARGET_COLORS = {
    0: "#2563EB",
    1: "#DC2626",
}


def plot_target_distribution(
    target_distribution: pd.DataFrame,
    *,
    class_labels: dict[int, str] | None = None,
) -> go.Figure:
    """Return a combined bar and pie chart for a binary target distribution."""
    labels = class_labels or DEFAULT_TARGET_LABELS
    frame = target_distribution.copy()
    frame["class_label"] = frame["class"].map(labels).fillna(frame["class"].astype(str))
    frame["rate_percent"] = frame["rate"] * 100
    frame = frame.sort_values("class")
    color_values = [TARGET_COLORS.get(value, NEUTRAL_COLOR) for value in frame["class"]]

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "bar"}, {"type": "domain"}]],
        column_widths=[0.58, 0.42],
        subplot_titles=("Rows by class", "Class share"),
    )
    figure.add_trace(
        go.Bar(
            x=frame["class_label"],
            y=frame["rows"],
            marker_color=color_values,
            text=frame["rate_percent"].map(lambda value: f"{value:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Rows=%{y}<br>Rate=%{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Pie(
            labels=frame["class_label"],
            values=frame["rows"],
            marker={"colors": color_values},
            textinfo="label+percent",
            hole=0.45,
            hovertemplate="<b>%{label}</b><br>Rows=%{value}<br>Share=%{percent}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_layout(
        title={
            "text": "Target distribution",
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        height=480,
        showlegend=False,
        margin={"l": 40, "r": 30, "t": 90, "b": 40},
        bargap=0.35,
    )
    figure.update_yaxes(title_text="Rows", row=1, col=1)
    figure.update_xaxes(title_text="Class", row=1, col=1)
    return figure


def plot_numeric_outlier_overview(
    frame: pd.DataFrame,
    features: list[str],
    *,
    target_column: str | None = None,
    max_features: int = 6,
) -> go.Figure:
    """Return histogram and boxplot panels for numeric outlier inspection."""
    selected = features[:max_features]
    if not selected:
        raise ValueError("At least one numeric feature is required")
    rows = len(selected)
    figure = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=[
            title
            for feature in selected
            for title in (f"{feature} distribution", f"{feature} boxplot")
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.08,
    )
    for index, feature in enumerate(selected, start=1):
        if target_column and target_column in frame.columns:
            for target_value, group in frame.groupby(target_column, dropna=False):
                color = TARGET_COLORS.get(target_value, NEUTRAL_COLOR)
                figure.add_trace(
                    go.Histogram(
                        x=group[feature],
                        histnorm="probability density",
                        name=f"{target_column}={target_value}",
                        legendgroup=str(target_value),
                        marker_color=color,
                        opacity=0.55,
                        showlegend=index == 1,
                    ),
                    row=index,
                    col=1,
                )
                figure.add_trace(
                    go.Box(
                        x=group[feature],
                        name=f"{target_column}={target_value}",
                        legendgroup=str(target_value),
                        marker_color=color,
                        boxmean=True,
                        showlegend=False,
                    ),
                    row=index,
                    col=2,
                )
        else:
            figure.add_trace(
                go.Histogram(
                    x=frame[feature],
                    histnorm="probability density",
                    marker_color=NEUTRAL_COLOR,
                    opacity=0.65,
                    showlegend=False,
                ),
                row=index,
                col=1,
            )
            figure.add_trace(
                go.Box(
                    x=frame[feature],
                    marker_color=NEUTRAL_COLOR,
                    boxmean=True,
                    name=feature,
                    showlegend=False,
                ),
                row=index,
                col=2,
            )
    figure.update_layout(
        title={"text": "Numeric feature outlier inspection", "x": 0.5},
        template="plotly_white",
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        height=max(420, 260 * rows),
        barmode="overlay",
        margin={"l": 50, "r": 30, "t": 90, "b": 40},
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor=GRID_COLOR, griddash="dot")
    return figure


def plot_categorical_feature_overview(
    frame: pd.DataFrame,
    features: list[str],
    *,
    target_column: str | None = None,
    max_features: int = 4,
    top_n: int = 8,
) -> go.Figure:
    """Return grouped frequency bars for categorical feature inspection."""
    selected = features[:max_features]
    if not selected:
        raise ValueError("At least one categorical feature is required")
    figure = make_subplots(
        rows=len(selected),
        cols=1,
        subplot_titles=[f"{feature} category frequency" for feature in selected],
        vertical_spacing=0.12,
    )
    for row, feature in enumerate(selected, start=1):
        work = frame[
            [feature] + ([target_column] if target_column in frame else [])
        ].copy()
        work[feature] = work[feature].astype("string").fillna("<missing>")
        top_categories = work[feature].value_counts().head(top_n).index.tolist()
        work[feature] = work[feature].where(work[feature].isin(top_categories), "Other")
        if target_column and target_column in work.columns:
            grouped = (
                work.groupby([feature, target_column], dropna=False)
                .size()
                .reset_index(name="rows")
            )
            for color_index, (target_value, group) in enumerate(
                grouped.groupby(target_column, dropna=False)
            ):
                figure.add_trace(
                    go.Bar(
                        x=group[feature],
                        y=group["rows"],
                        name=f"{target_column}={target_value}",
                        marker_color=TARGET_COLORS.get(target_value, NEUTRAL_COLOR),
                        showlegend=row == 1,
                    ),
                    row=row,
                    col=1,
                )
        else:
            counts = work[feature].value_counts().reset_index()
            counts.columns = [feature, "rows"]
            figure.add_trace(
                go.Bar(
                    x=counts[feature],
                    y=counts["rows"],
                    marker_color=NEUTRAL_COLOR,
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
    figure.update_layout(
        title={"text": "Categorical feature quality overview", "x": 0.5},
        template="plotly_white",
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        height=max(420, 280 * len(selected)),
        barmode="group",
        margin={"l": 50, "r": 30, "t": 90, "b": 60},
    )
    figure.update_xaxes(tickangle=-25)
    figure.update_yaxes(title_text="Rows", gridcolor=GRID_COLOR, griddash="dot")
    return figure


class DataQualityVisualizer:
    """Render data-quality visuals for one inspected dataset."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str | None = None,
    ):
        if frame is None or frame.empty:
            raise ValueError("DataQualityVisualizer requires a non-empty DataFrame")
        self.frame = frame
        self.target_column = target_column

    def numeric_outlier_overview(
        self,
        features: list[str],
        *,
        max_features: int = 6,
    ) -> go.Figure:
        """Return histogram and boxplot panels for numeric outlier inspection."""
        return plot_numeric_outlier_overview(
            self.frame,
            features,
            target_column=self.target_column,
            max_features=max_features,
        )

    def categorical_feature_overview(
        self,
        features: list[str],
        *,
        max_features: int = 4,
        top_n: int = 8,
    ) -> go.Figure:
        """Return grouped frequency bars for categorical feature inspection."""
        return plot_categorical_feature_overview(
            self.frame,
            features,
            target_column=self.target_column,
            max_features=max_features,
            top_n=top_n,
        )
