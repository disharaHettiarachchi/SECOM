"""Prediction and decision-support helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import get_prediction_scores
from src.utils import TARGET_LABELS, risk_band, risk_message


def _probability_from_scores(scores: np.ndarray) -> np.ndarray:
    """Convert raw decision scores to a 0-1 range when probabilities are absent."""

    if scores.min() >= 0 and scores.max() <= 1:
        return scores
    return 1 / (1 + np.exp(-scores))


def predict_faults(artifact: dict, records: pd.DataFrame) -> pd.DataFrame:
    """Predict normal/fault condition for one or more sensor records."""

    pipeline = artifact["pipeline"]
    predictions = pipeline.predict(records)
    scores = get_prediction_scores(pipeline, records)

    if scores is None:
        probabilities = predictions.astype(float)
    else:
        probabilities = _probability_from_scores(np.asarray(scores, dtype=float))

    output = records.copy()
    output["predicted_target"] = predictions.astype(int)
    output["predicted_condition"] = [TARGET_LABELS[int(value)] for value in predictions]
    output["fault_probability"] = probabilities
    output["risk_band"] = [risk_band(float(value)) for value in probabilities]
    output["decision_message"] = [risk_message(float(value)) for value in probabilities]
    return output


def sample_record_for_prediction(
    data: pd.DataFrame,
    feature_columns: list[str],
    row_index: int,
) -> pd.DataFrame:
    """Return one selected row in the format expected by the model."""

    return data.loc[[row_index], feature_columns]


def quality_recommendations(risk: str) -> list[str]:
    """Maintenance and quality-control recommendations for a risk band."""

    if risk == "High risk":
        return [
            "Hold or recheck the production item before final acceptance.",
            "Inspect the top contributing sensors for abnormal drift or unusual values.",
            "Review recent machine settings, maintenance logs, and process changes.",
            "Escalate repeated high-risk records to the process engineering team.",
        ]
    if risk == "Medium risk":
        return [
            "Run additional quality checks before releasing the batch.",
            "Compare sensor values with recent normal/pass records.",
            "Monitor the same process line for repeated medium-risk patterns.",
        ]
    return [
        "Continue normal production monitoring.",
        "Keep the prediction and sensor record for traceability.",
        "Re-train the model periodically as new labelled production data becomes available.",
    ]


def high_risk_records(
    artifact: dict,
    data: pd.DataFrame,
    feature_columns: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Score all records and return the highest-risk examples."""

    predictions = predict_faults(artifact, data[feature_columns])
    output = data[["timestamp", "condition", "target"]].join(
        predictions[["predicted_condition", "fault_probability", "risk_band"]]
    )
    return (
        output.sort_values("fault_probability", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

