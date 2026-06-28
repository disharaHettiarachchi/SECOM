"""Streamlit page for important sensors and feature distributions."""

import streamlit as st

from src.data_loader import load_secom_dataset, split_features_and_target
from src.feature_selection import model_feature_importance, target_correlation
from src.model_training import load_model_artifact
from src.visualizations import plot_feature_importance, plot_sensor_distribution


st.set_page_config(page_title="Sensor Insights", layout="wide")


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


st.title("Sensor Insights")

data = cached_dataset()
features, target = split_features_and_target(data)
artifact = get_or_load_artifact()

top_n = st.sidebar.slider("Top sensors", 5, 40, 20, 5)

st.subheader("Model-Based Feature Importance")
if artifact is None:
    st.info("Train a model to view model-based feature importance.")
else:
    importance = model_feature_importance(artifact["pipeline"], top_n=top_n)
    if importance.empty:
        st.warning("The active model does not expose feature importance values.")
    else:
        st.caption(f"Active model: {artifact['model_name']}")
        st.plotly_chart(plot_feature_importance(importance), use_container_width=True)
        st.dataframe(importance, use_container_width=True, hide_index=True)

st.subheader("Target-Correlation Sensor Ranking")
correlations = target_correlation(features, target, top_n=top_n)
left_column, right_column = st.columns([1, 1])
with left_column:
    st.dataframe(
        correlations,
        use_container_width=True,
        hide_index=True,
        column_config={
            "feature": "Sensor",
            "correlation": st.column_config.NumberColumn("Correlation", format="%.4f"),
            "absolute_correlation": st.column_config.NumberColumn(
                "Absolute correlation",
                format="%.4f",
            ),
        },
    )

with right_column:
    selected_feature = st.selectbox(
        "Sensor distribution",
        correlations["feature"].tolist(),
    )
    st.plotly_chart(
        plot_sensor_distribution(data, selected_feature),
        use_container_width=True,
    )

st.info(
    "Feature importance and correlation are decision-support indicators. They help "
    "identify sensor signals worth investigating, but they do not prove direct causation."
)

