"""Feature-ranking helpers used by the Sensor Insights dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_selected_feature_names(pipeline) -> list[str]:
    """Read feature names after cleaning and optional SelectKBest."""

    cleaner = pipeline.named_steps.get("cleaner")
    if cleaner is not None and hasattr(cleaner, "get_feature_names_out"):
        feature_names = list(cleaner.get_feature_names_out())
    else:
        feature_names = list(getattr(pipeline, "feature_names_in_", []))

    selector = pipeline.named_steps.get("selector")
    if selector is not None and hasattr(selector, "get_support"):
        mask = selector.get_support()
        if len(mask) == len(feature_names):
            feature_names = [
                feature for feature, keep_feature in zip(feature_names, mask) if keep_feature
            ]

    return feature_names


def model_feature_importance(pipeline, top_n: int = 20) -> pd.DataFrame:
    """Extract feature importance, coefficients, or selector scores."""

    feature_names = get_selected_feature_names(pipeline)
    model = pipeline.named_steps.get("model")
    selector = pipeline.named_steps.get("selector")
    importance_type = "model score"

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        importance_type = "tree importance"
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
        importance_type = "absolute coefficient"
    elif selector is not None and hasattr(selector, "scores_"):
        support = selector.get_support()
        values = np.asarray(selector.scores_)[support]
        importance_type = "ANOVA selector score"
    else:
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    if len(values) != len(feature_names):
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
            "importance_type": importance_type,
        }
    )
    return (
        importance.sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def target_correlation(features: pd.DataFrame, target: pd.Series, top_n: int = 20) -> pd.DataFrame:
    """Rank sensors by absolute Pearson correlation with the binary target."""

    filled = features.copy()
    filled = filled.fillna(filled.median(numeric_only=True))

    correlations = filled.corrwith(target).replace([np.inf, -np.inf], np.nan).dropna()
    correlation_table = pd.DataFrame(
        {
            "feature": correlations.index,
            "correlation": correlations.values,
            "absolute_correlation": correlations.abs().values,
        }
    )
    return (
        correlation_table.sort_values("absolute_correlation", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

