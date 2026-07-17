"""Training routines for the industrial fault detection models."""

from __future__ import annotations

import time
import platform
from dataclasses import asdict, dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluation import comparison_table, evaluate_classifier
from src.preprocessing import FeatureCleaner, PreprocessingConfig
from src.utils import MODELS_DIR


@dataclass
class TrainingConfig:
    """User-adjustable model-training settings."""

    test_size: float = 0.25
    random_state: int = 42
    missing_threshold: float = 0.50
    selected_features: int = 80
    imputation_strategy: str = "median"
    include_xgboost: bool = False
    selection_metric: str = "f1"
    decision_threshold: float = 0.50


def available_models(random_state: int = 42, include_xgboost: bool = False) -> dict[str, Any]:
    """Define the model set used for comparison."""

    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Support Vector Machine": SVC(
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }

    if include_xgboost:
        try:
            from xgboost import XGBClassifier

            models["XGBoost"] = XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=random_state,
            )
        except Exception:
            # XGBoost is optional. The required project still has four sklearn models.
            pass

    return models


def _selected_feature_count(X_train: pd.DataFrame, config: TrainingConfig) -> int:
    """Keep SelectKBest from requesting more features than are available."""

    cleaner = FeatureCleaner(missing_threshold=config.missing_threshold)
    cleaner.fit(X_train)
    kept_count = len(cleaner.features_kept_)
    return max(1, min(config.selected_features, kept_count))


def build_pipeline(model, config: TrainingConfig, k_best: int) -> Pipeline:
    """Build one complete modelling pipeline."""

    preprocessing_config = PreprocessingConfig(
        missing_threshold=config.missing_threshold,
        imputation_strategy=config.imputation_strategy,
        selected_features=k_best,
    )

    steps: list[tuple[str, Any]] = [
        ("cleaner", FeatureCleaner(preprocessing_config.missing_threshold)),
        ("imputer", SimpleImputer(strategy=preprocessing_config.imputation_strategy)),
        ("scaler", StandardScaler()),
    ]

    if k_best > 0:
        steps.append(("selector", SelectKBest(score_func=f_classif, k=k_best)))

    steps.append(("model", model))
    return Pipeline(steps)


def train_and_evaluate_models(
    features: pd.DataFrame,
    target: pd.Series,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Train all models and return metrics, fitted pipelines, and split details."""

    config = config or TrainingConfig()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target.astype(int),
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )

    k_best = _selected_feature_count(X_train, config)
    results: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for model_name, estimator in available_models(
        config.random_state,
        include_xgboost=config.include_xgboost,
    ).items():
        pipeline = build_pipeline(estimator, config, k_best)
        started_at = time.perf_counter()

        try:
            if model_name in {"Gradient Boosting", "XGBoost"}:
                sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
                pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
            else:
                pipeline.fit(X_train, y_train)

            training_seconds = time.perf_counter() - started_at
            evaluation = evaluate_classifier(pipeline, X_test, y_test)
            results[model_name] = {
                "evaluation": evaluation,
                "training_seconds": float(training_seconds),
                "error": None,
            }
            fitted_models[model_name] = pipeline
        except Exception as exc:
            results[model_name] = {
                "evaluation": None,
                "training_seconds": None,
                "error": str(exc),
            }

    table = comparison_table(results)
    successful_table = table[table["status"] == "trained"] if not table.empty else table
    best_model_name = None
    if not successful_table.empty:
        selection_metric = config.selection_metric
        if selection_metric not in successful_table.columns:
            raise ValueError(f"Unknown model-selection metric: {selection_metric}")
        ranked_models = successful_table.sort_values(
            by=[selection_metric, "recall", "pr_auc", "balanced_accuracy"],
            ascending=False,
            na_position="last",
        )
        best_model_name = str(ranked_models.iloc[0]["model"])

    return {
        "config": asdict(config),
        "model_results": results,
        "comparison_table": table,
        "fitted_models": fitted_models,
        "best_model_name": best_model_name,
        "feature_columns": list(features.columns),
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train": y_train,
        "y_test": y_test,
        "X_test": X_test,
        "selected_feature_count": k_best,
        "selection_metric": config.selection_metric,
    }


def create_model_artifact(training_result: dict[str, Any]) -> dict[str, Any]:
    """Package the best trained model and useful metadata for Joblib."""

    best_model_name = training_result["best_model_name"]
    if best_model_name is None:
        raise ValueError("No model was trained successfully.")

    all_model_metrics = {}
    for model_name, result in training_result["model_results"].items():
        if result.get("error"):
            all_model_metrics[model_name] = {"error": result["error"]}
            continue
        all_model_metrics[model_name] = {
            "metrics": result["evaluation"]["metrics"],
            "training_seconds": result["training_seconds"],
        }

    return {
        "model_name": best_model_name,
        "pipeline": training_result["fitted_models"][best_model_name],
        "metrics": training_result["model_results"][best_model_name]["evaluation"]["metrics"],
        "best_evaluation": training_result["model_results"][best_model_name]["evaluation"],
        "all_model_metrics": all_model_metrics,
        "comparison_table": training_result["comparison_table"],
        "config": training_result["config"],
        "feature_columns": training_result.get("feature_columns"),
        "training_summary": {
            "train_records": int(training_result["X_train_shape"][0]),
            "test_records": int(training_result["X_test_shape"][0]),
        },
        "selected_feature_count": training_result["selected_feature_count"],
        "selection_metric": training_result.get("selection_metric", "f1"),
        "decision_threshold": float(training_result["config"].get("decision_threshold", 0.50)),
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "artifact_note": "Generated from the Colab/local training workflow for Streamlit Cloud inference.",
    }


def validate_model_artifact(artifact: dict[str, Any]) -> None:
    """Raise a clear error when an uploaded file is not a usable artifact."""

    required_keys = {"model_name", "pipeline", "metrics", "feature_columns"}
    missing_keys = sorted(required_keys.difference(artifact))
    if missing_keys:
        raise ValueError(
            "The Joblib file is missing required fields: " + ", ".join(missing_keys)
        )


def artifact_runtime_status(artifact: dict[str, Any]) -> dict[str, str | bool | None]:
    """Describe whether the current scikit-learn runtime matches training."""

    versions = artifact.get("library_versions", {})
    trained_version = versions.get("scikit_learn")
    runtime_version = sklearn.__version__
    compatible = trained_version is None or trained_version == runtime_version
    return {
        "trained_sklearn": trained_version,
        "runtime_sklearn": runtime_version,
        "compatible": compatible,
    }


def save_model_artifact(
    artifact: dict[str, Any],
    filename: str = "secom_fault_detection_model.joblib",
) -> str:
    """Save the chosen model artifact in the models folder."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / filename
    joblib.dump(artifact, output_path)
    return str(output_path)


def load_model_artifact(
    filename: str = "secom_fault_detection_model.joblib",
) -> dict[str, Any] | None:
    """Load a saved model artifact if it exists."""

    path = MODELS_DIR / filename
    if not path.exists():
        return None
    artifact = joblib.load(path)
    validate_model_artifact(artifact)
    return artifact


def load_uploaded_model_artifact(uploaded_file) -> dict[str, Any]:
    """Load a Joblib model artifact uploaded through Streamlit."""

    artifact = joblib.load(uploaded_file)
    validate_model_artifact(artifact)
    return artifact
