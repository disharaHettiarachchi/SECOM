"""Common project paths and small helper functions.

The Streamlit pages and source modules import these values so the project keeps
one consistent folder layout on local machines and Streamlit Cloud.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_SENSOR_FILE = "secom.data"
RAW_LABEL_FILE = "secom_labels.data"
RAW_NAMES_FILE = "secom.names"

NORMAL_CLASS = 0
FAULT_CLASS = 1

TARGET_LABELS = {
    NORMAL_CLASS: "Normal / Pass",
    FAULT_CLASS: "Fault / Fail",
}

RAW_LABEL_MAPPING = {
    -1: NORMAL_CLASS,
    1: FAULT_CLASS,
}


def ensure_project_directories() -> None:
    """Create project folders that may be missing after a fresh clone."""

    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        ASSETS_DIR,
        DOCS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a fraction such as 0.1234 as a human-readable percentage."""

    return f"{value * 100:.{decimals}f}%"


def condition_name(target_value: int) -> str:
    """Convert the internal binary target into a dashboard label."""

    return TARGET_LABELS.get(int(target_value), "Unknown")


def risk_band(probability: float) -> str:
    """Convert a fault probability into an easy decision-support band."""

    if probability >= 0.70:
        return "High risk"
    if probability >= 0.40:
        return "Medium risk"
    return "Low risk"


def risk_message(probability: float) -> str:
    """Short explanation shown beside individual predictions."""

    band = risk_band(probability)
    if band == "High risk":
        return "The record shows a strong fault signal. Prioritise inspection before accepting the production batch."
    if band == "Medium risk":
        return "The record has warning signs. Review the key sensor values and consider additional quality checks."
    return "The record is currently similar to normal/pass examples, but routine monitoring should continue."

