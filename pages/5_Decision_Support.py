"""Streamlit page for operational decision-support outputs."""

import pandas as pd
import streamlit as st

from src.data_loader import get_feature_columns, load_secom_dataset
from src.model_training import load_model_artifact, load_uploaded_model_artifact
from src.prediction import high_risk_records, quality_recommendations
from src.visualizations import plot_risk_records


st.set_page_config(page_title="Decision Support", layout="wide")


@st.cache_data(show_spinner=False)
def cached_dataset():
    return load_secom_dataset()


def get_or_load_artifact():
    if "model_artifact" in st.session_state:
        return st.session_state.model_artifact
    artifact = load_model_artifact()
    if artifact:
        st.session_state.model_artifact = artifact
    return artifact


st.title("Decision Support")

data = cached_dataset()
feature_columns = get_feature_columns(data)
artifact = get_or_load_artifact()

if artifact is None:
    st.warning("A trained model artifact is required for high-risk record ranking.")
    uploaded_artifact = st.file_uploader(
        "Upload `secom_fault_detection_model.joblib` exported from Colab",
        type=["joblib"],
    )
    if uploaded_artifact is not None:
        try:
            artifact = load_uploaded_model_artifact(uploaded_artifact)
            st.session_state.model_artifact = artifact
            st.success(f"Uploaded model loaded: {artifact['model_name']}")
        except Exception as exc:
            st.error(f"Could not load the uploaded model: {exc}")
            st.stop()
    if artifact is None:
        st.info(
            "Run the Colab training notebook and place the exported model artifact "
            "in `models/` before deploying, or upload it here during a demo."
        )
        st.stop()

st.success(f"Active model: {artifact['model_name']}")

top_n = st.slider("High-risk records to display", 5, 50, 20, 5)
ranked_records = high_risk_records(artifact, data, feature_columns, top_n=top_n)

st.subheader("Highest-Risk Production Records")
st.plotly_chart(plot_risk_records(ranked_records), use_container_width=True)
st.dataframe(
    ranked_records,
    use_container_width=True,
    hide_index=True,
    column_config={
        "timestamp": "Timestamp",
        "condition": "Actual condition",
        "target": "Actual target",
        "predicted_condition": "Predicted condition",
        "fault_probability": st.column_config.ProgressColumn(
            "Fault probability",
            min_value=0.0,
            max_value=1.0,
            format="%.3f",
        ),
        "risk_band": "Risk band",
    },
)

st.subheader("Action Matrix")
action_rows = []
for band in ["High risk", "Medium risk", "Low risk"]:
    action_rows.append(
        {
            "risk_band": band,
            "recommended_actions": " | ".join(quality_recommendations(band)),
        }
    )

st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)

st.warning(
    "Predictions should support, not replace, engineering judgement. A high-risk "
    "score means the production record should be prioritised for inspection and "
    "process review."
)
