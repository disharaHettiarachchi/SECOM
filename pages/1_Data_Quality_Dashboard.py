"""Streamlit page for missing values, removed features, and imbalance."""

import streamlit as st

from src.data_loader import load_secom_dataset, missing_value_summary, split_features_and_target
from src.preprocessing import build_feature_quality_report, class_distribution
from src.visualizations import (
    plot_class_distribution,
    plot_feature_quality_status,
    plot_missing_values,
)


st.set_page_config(page_title="Data Quality Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def cached_dataset():
    return load_secom_dataset()


st.title("Data Quality Dashboard")

data = cached_dataset()
features, target = split_features_and_target(data)

missing_threshold = st.sidebar.slider(
    "High-missing feature threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)

quality_report, quality_counts = build_feature_quality_report(
    features,
    missing_threshold=missing_threshold,
)
missing_summary = missing_value_summary(features)
class_summary = class_distribution(target)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Original features", quality_counts["original_features"])
metric_2.metric("Removed features", quality_counts["total_removed"])
metric_3.metric("Kept features", quality_counts["features_kept"])
metric_4.metric("Imputation method", "Median")

left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Feature Missingness")
    st.plotly_chart(plot_missing_values(missing_summary, top_n=30), use_container_width=True)

with right_column:
    st.subheader("Feature Removal Status")
    st.plotly_chart(plot_feature_quality_status(quality_report), use_container_width=True)

st.subheader("Removed Features")
removed_features = quality_report[quality_report["status"] != "kept"].copy()
st.dataframe(
    removed_features.sort_values(["status", "missing_percentage"], ascending=[True, False]),
    use_container_width=True,
    hide_index=True,
    column_config={
        "feature": "Sensor",
        "missing_percentage": st.column_config.NumberColumn("Missing (%)", format="%.2f"),
        "unique_values": "Unique values",
        "status": "Removal reason",
    },
)

st.subheader("Class Imbalance")
imbalance_left, imbalance_right = st.columns([0.9, 1.1])
with imbalance_left:
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
with imbalance_right:
    st.plotly_chart(plot_class_distribution(data), use_container_width=True)

st.info(
    "During training, class imbalance is handled with class weights for Logistic "
    "Regression, Random Forest, and SVM, and balanced sample weights for Gradient Boosting."
)

