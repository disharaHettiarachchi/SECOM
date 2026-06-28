"""Plotly visualisations used across the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLOR_NORMAL = "#2E7D32"
COLOR_FAULT = "#C62828"
COLOR_ACCENT = "#1565C0"


def plot_class_distribution(data: pd.DataFrame) -> go.Figure:
    counts = data["condition"].value_counts().reset_index()
    counts.columns = ["condition", "count"]
    fig = px.bar(
        counts,
        x="condition",
        y="count",
        color="condition",
        color_discrete_map={
            "Normal / Pass": COLOR_NORMAL,
            "Fault / Fail": COLOR_FAULT,
        },
        text="count",
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Records")
    return fig


def plot_missing_values(missing_summary: pd.DataFrame, top_n: int = 30) -> go.Figure:
    top_missing = missing_summary.head(top_n).sort_values("missing_percentage")
    fig = px.bar(
        top_missing,
        x="missing_percentage",
        y="feature",
        orientation="h",
        labels={"missing_percentage": "Missing values (%)", "feature": "Sensor"},
        color_discrete_sequence=[COLOR_ACCENT],
    )
    fig.update_layout(height=max(420, top_n * 18), yaxis_title="")
    return fig


def plot_feature_quality_status(report: pd.DataFrame) -> go.Figure:
    status_counts = report["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    fig = px.bar(
        status_counts,
        x="status",
        y="count",
        text="count",
        labels={"status": "Feature status", "count": "Number of features"},
        color="status",
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_model_comparison(comparison: pd.DataFrame, metric: str = "f1") -> go.Figure:
    trained = comparison[comparison["status"] == "trained"].copy()
    fig = px.bar(
        trained,
        x="model",
        y=metric,
        color="model",
        text=metric,
        labels={"model": "Model", metric: metric.replace("_", " ").title()},
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, 1])
    return fig


def plot_confusion_matrix(matrix) -> go.Figure:
    labels = ["Normal / Pass", "Fault / Fail"]
    fig = px.imshow(
        matrix,
        x=[f"Predicted {label}" for label in labels],
        y=[f"Actual {label}" for label in labels],
        text_auto=True,
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def plot_roc_curve(curves: dict[str, object], roc_auc: float | None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curves.get("fpr", []),
            y=curves.get("tpr", []),
            mode="lines",
            name=f"ROC curve (AUC={roc_auc:.3f})" if roc_auc is not None else "ROC curve",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="Random baseline",
        )
    )
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig


def plot_pr_curve(curves: dict[str, object], pr_auc: float | None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curves.get("recall_curve", []),
            y=curves.get("precision_curve", []),
            mode="lines",
            name=f"PR curve (AP={pr_auc:.3f})" if pr_auc is not None else "PR curve",
        )
    )
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", yaxis_range=[0, 1])
    return fig


def plot_feature_importance(importance: pd.DataFrame) -> go.Figure:
    if importance.empty:
        return go.Figure()

    chart_data = importance.sort_values("importance")
    fig = px.bar(
        chart_data,
        x="importance",
        y="feature",
        orientation="h",
        color_discrete_sequence=[COLOR_ACCENT],
        labels={"importance": "Importance", "feature": "Sensor"},
    )
    fig.update_layout(height=max(420, len(chart_data) * 22), yaxis_title="")
    return fig


def plot_sensor_distribution(
    data: pd.DataFrame,
    feature: str,
) -> go.Figure:
    fig = px.histogram(
        data,
        x=feature,
        color="condition",
        marginal="box",
        nbins=40,
        barmode="overlay",
        color_discrete_map={
            "Normal / Pass": COLOR_NORMAL,
            "Fault / Fail": COLOR_FAULT,
        },
    )
    fig.update_layout(yaxis_title="Records")
    return fig


def plot_risk_records(records: pd.DataFrame) -> go.Figure:
    chart_data = records.reset_index().rename(columns={"index": "record_rank"})
    fig = px.bar(
        chart_data,
        x="record_rank",
        y="fault_probability",
        color="risk_band",
        hover_data=["timestamp", "condition", "predicted_condition"],
        labels={"fault_probability": "Fault probability", "record_rank": "Record rank"},
    )
    fig.update_layout(yaxis_range=[0, 1])
    return fig

