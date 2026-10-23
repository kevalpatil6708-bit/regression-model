from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from mlproject.config import (
    CLASSIFICATION_MODEL_FILE,
    REGRESSION_MODEL_FILE,
)


def load_artifact(model_file: Path) -> dict:
    path = Path(model_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python -m mlproject train` first."
        )
    return joblib.load(path)


def _as_frame(records: pd.DataFrame | list[dict] | dict) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, dict):
        return pd.DataFrame([records])
    return pd.DataFrame(records)


def _select_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return frame[features]


def predict_regression(
    records: pd.DataFrame | list[dict] | dict,
    model_file: Path = REGRESSION_MODEL_FILE,
) -> pd.DataFrame:
    artifact = load_artifact(model_file)
    frame = _select_features(_as_frame(records), artifact["features"])
    output = records.copy() if isinstance(records, pd.DataFrame) else _as_frame(records)
    output["predicted_energy_kwh"] = artifact["model"].predict(frame)
    return output


def predict_classification(
    records: pd.DataFrame | list[dict] | dict,
    model_file: Path = CLASSIFICATION_MODEL_FILE,
    threshold: float = 0.5,
) -> pd.DataFrame:
    artifact = load_artifact(model_file)
    frame = _select_features(_as_frame(records), artifact["features"])
    probabilities = artifact["model"].predict_proba(frame)[:, 1]
    output = records.copy() if isinstance(records, pd.DataFrame) else _as_frame(records)
    output["churn_probability"] = probabilities
    output["churn_prediction"] = (probabilities >= threshold).astype(int)
    return output
