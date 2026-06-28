"""Streamlit page for predicting normal/fault condition."""

import pandas as pd
import streamlit as st

from src.data_loader import get_feature_columns, load_secom_dataset
from src.model_training import load_model_artifact, load_uploaded_model_artifact
from src.prediction import (
    predict_faults,
    quality_recommendations,
    sample_record_for_prediction,
)


st.set_page_config(page_title="Fault Prediction", layout="wide")


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


def prepare_uploaded_data(uploaded_file, feature_columns: list[str]) -> pd.DataFrame:
    uploaded = pd.read_csv(uploaded_file)

    if set(feature_columns).issubset(uploaded.columns):
        return uploaded[feature_columns]

    if uploaded.shape[1] >= len(feature_columns):
        sensor_only = uploaded.iloc[:, : len(feature_columns)].copy()
        sensor_only.columns = feature_columns
        return sensor_only

    raise ValueError(
        f"Uploaded file must contain {len(feature_columns)} sensor columns or named sensor columns."
    )


st.title("Fault Prediction")

data = cached_dataset()
feature_columns = get_feature_columns(data)
artifact = get_or_load_artifact()

if artifact is None:
    st.warning("A trained model artifact is required before predictions can be produced.")
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
            "Run `notebooks/SECOM_Model_Training_Colab.ipynb`, then either commit "
            "`models/secom_fault_detection_model.joblib` to GitHub for Streamlit Cloud "
            "or upload it on this page during a demo."
        )
        st.stop()

st.success(f"Active model: {artifact['model_name']}")

prediction_mode = st.radio(
    "Prediction input",
    ["Select a SECOM sample record", "Upload new sensor CSV"],
    horizontal=True,
)

records_to_predict = None
actual_condition = None

if prediction_mode == "Select a SECOM sample record":
    selected_index = st.number_input(
        "Sample record index",
        min_value=0,
        max_value=len(data) - 1,
        value=0,
        step=1,
    )
    records_to_predict = sample_record_for_prediction(
        data,
        feature_columns,
        int(selected_index),
    )
    actual_condition = data.loc[int(selected_index), "condition"]
    st.caption(f"Known dataset label for this sample: {actual_condition}")
    with st.expander("Selected sensor values"):
        st.dataframe(records_to_predict, use_container_width=True)

else:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            records_to_predict = prepare_uploaded_data(uploaded_file, feature_columns)
            st.caption(f"Uploaded records: {len(records_to_predict)}")
        except ValueError as exc:
            st.error(str(exc))

if records_to_predict is not None and st.button("Predict fault risk", type="primary"):
    predictions = predict_faults(artifact, records_to_predict)
    result_columns = [
        "predicted_condition",
        "fault_probability",
        "risk_band",
        "decision_message",
    ]
    st.subheader("Prediction Result")
    st.dataframe(
        predictions[result_columns],
        use_container_width=True,
        column_config={
            "predicted_condition": "Prediction",
            "fault_probability": st.column_config.ProgressColumn(
                "Fault probability",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
            "risk_band": "Risk band",
            "decision_message": "Decision support message",
        },
    )

    first_risk = str(predictions.iloc[0]["risk_band"])
    st.subheader("Recommended Actions")
    for recommendation in quality_recommendations(first_risk):
        st.write(f"- {recommendation}")
