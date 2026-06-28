"""Functions for loading and describing the UCI SECOM dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    RAW_LABEL_FILE,
    RAW_LABEL_MAPPING,
    RAW_NAMES_FILE,
    RAW_SENSOR_FILE,
    TARGET_LABELS,
)


def _candidate_paths(filename: str) -> Iterable[Path]:
    """Search both the project root and data/raw for a dataset file."""

    yield RAW_DATA_DIR / filename
    yield PROJECT_ROOT / filename


def find_dataset_file(filename: str) -> Path:
    """Return the first existing path for a required dataset file."""

    for path in _candidate_paths(filename):
        if path.exists():
            return path
    searched = ", ".join(str(path) for path in _candidate_paths(filename))
    raise FileNotFoundError(f"Could not find {filename}. Searched: {searched}")


def sensor_column_names(number_of_columns: int) -> list[str]:
    """Create readable names such as sensor_000, sensor_001, and so on."""

    return [f"sensor_{index:03d}" for index in range(number_of_columns)]


def load_sensor_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the SECOM sensor matrix.

    The raw file is whitespace separated and uses the text value "NaN" for
    missing sensor readings.
    """

    sensor_path = Path(path) if path else find_dataset_file(RAW_SENSOR_FILE)
    sensor_data = pd.read_csv(
        sensor_path,
        sep=r"\s+",
        header=None,
        na_values=["NaN"],
        engine="python",
    )
    sensor_data.columns = sensor_column_names(sensor_data.shape[1])
    return sensor_data


def load_label_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load SECOM labels and timestamps.

    The label file stores one raw label and one quoted timestamp per row. The
    UCI convention is -1 for pass and 1 for fail; the app maps this to 0 for
    normal/pass and 1 for fault/fail so the positive class is the fault class.
    """

    label_path = Path(path) if path else find_dataset_file(RAW_LABEL_FILE)
    rows: list[dict[str, object]] = []

    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            clean_line = line.strip()
            if not clean_line:
                continue

            try:
                raw_label_text, timestamp_text = clean_line.split(maxsplit=1)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label row at line {line_number}: {clean_line!r}"
                ) from exc

            raw_label = int(raw_label_text)
            if raw_label not in RAW_LABEL_MAPPING:
                raise ValueError(f"Unexpected label {raw_label} at line {line_number}")

            rows.append(
                {
                    "raw_label": raw_label,
                    "target": RAW_LABEL_MAPPING[raw_label],
                    "timestamp": timestamp_text.strip().strip('"'),
                }
            )

    label_data = pd.DataFrame(rows)
    label_data["timestamp"] = pd.to_datetime(
        label_data["timestamp"],
        dayfirst=True,
        errors="coerce",
    )
    label_data["condition"] = label_data["target"].map(TARGET_LABELS)
    return label_data


def load_secom_dataset(
    sensor_path: str | Path | None = None,
    label_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load and combine sensor readings, binary target, and timestamp."""

    sensors = load_sensor_data(sensor_path)
    labels = load_label_data(label_path)

    if len(sensors) != len(labels):
        raise ValueError(
            "Sensor rows and label rows do not match: "
            f"{len(sensors)} sensor rows vs {len(labels)} label rows."
        )

    return pd.concat([labels, sensors], axis=1)


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """Return only the sensor feature columns from a combined dataset."""

    return [column for column in data.columns if column.startswith("sensor_")]


def split_features_and_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the model input matrix and the binary target column."""

    feature_columns = get_feature_columns(data)
    return data[feature_columns], data["target"].astype(int)


def dataset_summary(data: pd.DataFrame) -> dict[str, object]:
    """Build a compact summary used by the Project Overview page."""

    feature_columns = get_feature_columns(data)
    features = data[feature_columns]
    target_counts = data["target"].value_counts().sort_index()

    return {
        "records": int(len(data)),
        "features": int(len(feature_columns)),
        "normal_records": int(target_counts.get(0, 0)),
        "fault_records": int(target_counts.get(1, 0)),
        "fault_rate": float(target_counts.get(1, 0) / len(data)),
        "total_missing_values": int(features.isna().sum().sum()),
        "overall_missing_percentage": float(features.isna().mean().mean()),
        "date_min": data["timestamp"].min(),
        "date_max": data["timestamp"].max(),
    }


def missing_value_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages for every sensor feature."""

    summary = pd.DataFrame(
        {
            "feature": features.columns,
            "missing_count": features.isna().sum().values,
            "missing_percentage": features.isna().mean().values * 100,
        }
    )
    return summary.sort_values("missing_percentage", ascending=False).reset_index(drop=True)


def load_dataset_notes(path: str | Path | None = None) -> str:
    """Load the human-readable SECOM notes file if it is available."""

    names_path = Path(path) if path else find_dataset_file(RAW_NAMES_FILE)
    return names_path.read_text(encoding="utf-8", errors="ignore")


def save_combined_dataset(data: pd.DataFrame, path: str | Path) -> Path:
    """Save the combined dataset as CSV for inspection or report evidence."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path

