"""Main Streamlit entry point: Project Overview."""

import streamlit as st

from src.data_loader import (
    dataset_summary,
    load_dataset_notes,
    load_secom_dataset,
    missing_value_summary,
    split_features_and_target,
)
from src.preprocessing import class_distribution
from src.utils import ensure_project_directories, format_percentage
from src.visualizations import plot_class_distribution, plot_missing_values


st.set_page_config(
    page_title="Industrial Fault Detection System",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_dataset():
    return load_secom_dataset()


@st.cache_data(show_spinner=False)
def cached_dataset_notes():
    return load_dataset_notes()


ensure_project_directories()

st.title("Machine Learning-Based Industrial Fault Detection System")
st.caption("SECOM semiconductor sensor data | Imbalance-aware normal/fault classification")

data = cached_dataset()
features, target = split_features_and_target(data)
summary = dataset_summary(data)
missing_summary = missing_value_summary(features)
class_summary = class_distribution(target)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Records", f"{summary['records']:,}")
metric_2.metric("Sensor features", f"{summary['features']:,}")
metric_3.metric("Fault records", f"{summary['fault_records']:,}")
metric_4.metric("Fault rate", format_percentage(summary["fault_rate"]))

date_min = summary["date_min"].date() if summary["date_min"] is not None else "Unknown"
date_max = summary["date_max"].date() if summary["date_max"] is not None else "Unknown"

left_column, right_column = st.columns([1.1, 0.9])
with left_column:
    st.subheader("Dataset Summary")
    st.dataframe(
        class_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "target": "Target",
            "count": "Records",
            "percentage": st.column_config.NumberColumn("Percentage", format="%.2f%%"),
            "condition": "Condition",
        },
    )
    st.plotly_chart(plot_class_distribution(data), use_container_width=True)

with right_column:
    st.subheader("Missing Value Snapshot")
    st.metric(
        "Overall missing values",
        f"{summary['total_missing_values']:,}",
        format_percentage(summary["overall_missing_percentage"]),
    )
    st.metric("Data period", f"{date_min} to {date_max}")
    st.plotly_chart(plot_missing_values(missing_summary, top_n=12), use_container_width=True)

st.subheader("Project Framing")
st.markdown(
    """
This project treats the UCI SECOM dataset as a manufacturing quality-control
problem. Each production record contains sensor and process measurements from a
semiconductor manufacturing process, and the target label represents whether the
record passed or failed downstream testing.

The system follows a development-oriented final year project structure: data
preparation, model comparison, evaluation, dashboard implementation, prediction,
and decision-support reporting. The research component is the comparison of
machine learning models under class imbalance.
"""
)

st.subheader("System Workflow")
workflow_columns = st.columns(5)
for column, title, detail in zip(
    workflow_columns,
    ["1. Inspect", "2. Prepare", "3. Compare", "4. Predict", "5. Support"],
    [
        "Review missingness and class imbalance.",
        "Clean, impute, scale, and select features.",
        "Evaluate four supervised models.",
        "Score samples or uploaded sensor CSV files.",
        "Rank risk and recommend quality-control actions.",
    ],
):
    with column:
        st.markdown(f"**{title}**")
        st.caption(detail)

st.info(
    "Research prototype: predictions are decision-support signals based on an "
    "anonymised historical dataset. They do not replace process engineers or "
    "validated production quality procedures."
)

with st.expander("Dataset notes from secom.names"):
    notes = cached_dataset_notes()
    st.text(notes[:4000])
