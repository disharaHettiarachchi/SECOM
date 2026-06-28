"""Preprocessing utilities for SECOM sensor data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class PreprocessingConfig:
    """Configuration values used by the cleaner and model pipelines."""

    missing_threshold: float = 0.50
    imputation_strategy: str = "median"
    selected_features: int = 80


class FeatureCleaner(BaseEstimator, TransformerMixin):
    """Remove unusable sensor features before imputation and modelling.

    The transformer is fitted only on the training split when used inside a
    scikit-learn Pipeline. This prevents information from the test split from
    influencing feature-removal decisions.
    """

    def __init__(self, missing_threshold: float = 0.50):
        self.missing_threshold = missing_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        feature_data = self._to_dataframe(X)
        self.feature_names_in_ = list(feature_data.columns)

        missing_rates = feature_data.isna().mean()
        self.high_missing_features_ = missing_rates[
            missing_rates > self.missing_threshold
        ].index.tolist()

        remaining = feature_data.drop(columns=self.high_missing_features_, errors="ignore")
        unique_counts = remaining.nunique(dropna=True)
        self.constant_features_ = unique_counts[unique_counts <= 1].index.tolist()

        drop_set = set(self.high_missing_features_) | set(self.constant_features_)
        self.features_to_drop_ = [
            column for column in self.feature_names_in_ if column in drop_set
        ]
        self.features_kept_ = [
            column for column in self.feature_names_in_ if column not in drop_set
        ]

        if not self.features_kept_:
            raise ValueError("All features were removed. Increase the missing threshold.")

        self.removal_report_ = self._build_removal_report(feature_data)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        feature_data = self._to_dataframe(X)

        # Reindex keeps prediction robust when uploaded files contain columns in
        # a different order. Missing columns are filled with NaN and imputed later.
        feature_data = feature_data.reindex(columns=self.feature_names_in_)
        return feature_data[self.features_kept_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.features_kept_, dtype=object)

    def _to_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()

        if hasattr(self, "feature_names_in_"):
            columns = self.feature_names_in_
        else:
            columns = [f"sensor_{index:03d}" for index in range(np.asarray(X).shape[1])]
        return pd.DataFrame(X, columns=columns)

    def _build_removal_report(self, feature_data: pd.DataFrame) -> pd.DataFrame:
        missing_percentages = feature_data.isna().mean() * 100
        unique_counts = feature_data.nunique(dropna=True)
        reasons: list[str] = []

        for column in feature_data.columns:
            reason_parts: list[str] = []
            if column in self.high_missing_features_:
                reason_parts.append("high missing values")
            if column in self.constant_features_:
                reason_parts.append("constant or no useful variance")
            reasons.append(", ".join(reason_parts) if reason_parts else "kept")

        return pd.DataFrame(
            {
                "feature": feature_data.columns,
                "missing_percentage": missing_percentages.values,
                "unique_values": unique_counts.reindex(feature_data.columns).values,
                "status": reasons,
            }
        )


def build_feature_quality_report(
    features: pd.DataFrame,
    missing_threshold: float = 0.50,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Create a data-quality report without changing the original DataFrame."""

    cleaner = FeatureCleaner(missing_threshold=missing_threshold)
    cleaner.fit(features)
    report = cleaner.removal_report_.copy()
    counts = {
        "original_features": int(features.shape[1]),
        "removed_high_missing": int(len(cleaner.high_missing_features_)),
        "removed_constant": int(len(cleaner.constant_features_)),
        "total_removed": int(len(cleaner.features_to_drop_)),
        "features_kept": int(len(cleaner.features_kept_)),
    }
    return report, counts


def class_distribution(target: pd.Series) -> pd.DataFrame:
    """Summarise class imbalance for the dashboard."""

    counts = target.value_counts().sort_index()
    total = len(target)
    return pd.DataFrame(
        {
            "target": counts.index,
            "count": counts.values,
            "percentage": counts.values / total * 100,
            "condition": ["Normal / Pass" if value == 0 else "Fault / Fail" for value in counts.index],
        }
    )

