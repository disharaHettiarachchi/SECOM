"""Model evaluation helpers for imbalanced industrial fault detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def get_prediction_scores(model, features: pd.DataFrame) -> np.ndarray | None:
    """Return fault-class scores for metrics and risk displays."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.shape[1] > 1:
            return probabilities[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        return np.asarray(scores)

    return None


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    """Evaluate a fitted classifier with metrics suitable for imbalance."""

    y_pred = model.predict(X_test)
    y_score = get_prediction_scores(model, X_test)
    labels = [0, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
    }

    curves: dict[str, object] = {}
    if y_score is not None and len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_test, y_score))

        fpr, tpr, roc_thresholds = roc_curve(y_test, y_score)
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            y_test,
            y_score,
        )
        curves = {
            "fpr": fpr,
            "tpr": tpr,
            "roc_thresholds": roc_thresholds,
            "precision_curve": precision_curve,
            "recall_curve": recall_curve,
            "pr_thresholds": pr_thresholds,
        }
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=["Normal / Pass", "Fault / Fail"],
            output_dict=True,
            zero_division=0,
        ),
        "curves": curves,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def comparison_table(model_results: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Convert model results into an imbalance-aware comparison table.

    F1 and recall lead the ordering because a fault detector that misses every
    fault is not useful even when its overall accuracy or ranking AUC is high.
    """

    rows: list[dict[str, object]] = []
    for model_name, result in model_results.items():
        if result.get("error"):
            rows.append({"model": model_name, "status": result["error"]})
            continue

        metrics = result["evaluation"]["metrics"]
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "training_seconds": result.get("training_seconds"),
                "status": "trained",
            }
        )

    table = pd.DataFrame(rows)
    if "f1" in table.columns:
        table = table.sort_values(
            by=["f1", "recall", "pr_auc", "balanced_accuracy"],
            ascending=False,
            na_position="last",
        )
    return table.reset_index(drop=True)


def classification_report_table(report: dict[str, object]) -> pd.DataFrame:
    """Create a readable classification-report table."""

    table = pd.DataFrame(report).transpose()
    numeric_columns = table.select_dtypes(include="number").columns
    table[numeric_columns] = table[numeric_columns].round(4)
    return table
