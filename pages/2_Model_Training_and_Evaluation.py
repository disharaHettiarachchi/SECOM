"""Streamlit page for viewing model evaluation results from a saved artifact."""

import pandas as pd
import streamlit as st

from src.evaluation import classification_report_table
from src.model_training import load_model_artifact, load_uploaded_model_artifact
from src.visualizations import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pr_curve,
    plot_roc_curve,
)


st.set_page_config(page_title="Model Training and Evaluation", layout="wide")


def rounded_table(table: pd.DataFrame) -> pd.DataFrame:
    display_table = table.copy()
    numeric_columns = display_table.select_dtypes(include="number").columns
    display_table[numeric_columns] = display_table[numeric_columns].round(4)
    return display_table


st.title("Model Training and Evaluation")
st.caption("Training is expected to be run in Google Colab. This page reads the exported Joblib artifact.")

comparison_metric = st.sidebar.selectbox(
    "Comparison chart metric",
    ["f1", "recall", "precision", "balanced_accuracy", "roc_auc", "pr_auc"],
)

saved_artifact = load_model_artifact()
if saved_artifact and "model_artifact" not in st.session_state:
    st.session_state.model_artifact = saved_artifact

uploaded_artifact = st.file_uploader(
    "Optional: upload a Colab-exported Joblib artifact",
    type=["joblib"],
)
if uploaded_artifact is not None:
    try:
        st.session_state.model_artifact = load_uploaded_model_artifact(uploaded_artifact)
        st.success("Uploaded model artifact loaded for this session.")
    except Exception as exc:
        st.error(f"Could not load the uploaded artifact: {exc}")

artifact = st.session_state.get("model_artifact")
if artifact is None:
    st.info(
        "No model artifact was found. Run `notebooks/SECOM_Model_Training_Colab.ipynb`, "
        "download `secom_fault_detection_model.joblib`, then place it in `models/` "
        "or upload it here."
    )
    st.stop()

st.success(f"Active model artifact: {artifact['model_name']}")

comparison = artifact.get("comparison_table")
if comparison is None:
    comparison = pd.DataFrame(artifact.get("all_model_metrics", {})).transpose().reset_index()
    comparison = comparison.rename(columns={"index": "model"})
elif not isinstance(comparison, pd.DataFrame):
    comparison = pd.DataFrame(comparison)

st.subheader("Model Comparison")
st.dataframe(rounded_table(comparison), use_container_width=True, hide_index=True)

chart_column, detail_column = st.columns([1, 1])
with chart_column:
    if {"model", comparison_metric, "status"}.issubset(comparison.columns):
        st.plotly_chart(
            plot_model_comparison(comparison, metric=comparison_metric),
            use_container_width=True,
        )
    else:
        st.info("The artifact does not contain enough comparison-table data for this chart.")

with detail_column:
    training_summary = artifact.get("training_summary", {})
    st.metric("Best model", artifact.get("model_name", "Unknown"))
    st.metric("Selected features", artifact.get("selected_feature_count", "Unknown"))
    st.metric("Training split", f"{training_summary.get('train_records', 'Unknown')} records")
    st.metric("Test split", f"{training_summary.get('test_records', 'Unknown')} records")

selected_result = artifact.get("best_evaluation")
metrics = artifact.get("metrics", {})
if selected_result is None:
    st.warning("This artifact only contains summary metrics, not detailed curves/reports.")
    st.stop()

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Precision", f"{metrics['precision']:.3f}")
metric_2.metric("Recall", f"{metrics['recall']:.3f}")
metric_3.metric("F1-score", f"{metrics['f1']:.3f}")
metric_4.metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.3f}")

matrix_column, report_column = st.columns([0.9, 1.1])
with matrix_column:
    st.subheader("Confusion Matrix")
    st.plotly_chart(
        plot_confusion_matrix(selected_result["confusion_matrix"]),
        use_container_width=True,
    )

with report_column:
    st.subheader("Classification Report")
    st.dataframe(
        classification_report_table(selected_result["classification_report"]),
        use_container_width=True,
    )

curve_left, curve_right = st.columns(2)
with curve_left:
    st.subheader("ROC Curve")
    st.plotly_chart(
        plot_roc_curve(selected_result["curves"], metrics["roc_auc"]),
        use_container_width=True,
    )
with curve_right:
    st.subheader("Precision-Recall Curve")
    st.plotly_chart(
        plot_pr_curve(selected_result["curves"], metrics["pr_auc"]),
        use_container_width=True,
    )

failed = comparison[comparison["status"] != "trained"] if "status" in comparison.columns else pd.DataFrame()
if not failed.empty:
    with st.expander("Model training errors"):
        st.dataframe(failed, use_container_width=True, hide_index=True)
