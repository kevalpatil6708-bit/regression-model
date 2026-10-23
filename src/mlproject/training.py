from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlproject.config import ARTIFACTS_DIR
from mlproject.data import CLASSIFICATION_TARGET, REGRESSION_TARGET

REGRESSION_FEATURES = [
    "size_sqm",
    "occupants",
    "insulation_rating",
    "avg_temp_c",
    "building_age_years",
    "has_solar",
    "property_type",
]
CLASSIFICATION_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "contract_type",
    "internet_service",
    "payment_method",
    "satisfaction_score",
    "has_partner",
]

REGRESSION_CATEGORICAL = ["property_type"]
CLASSIFICATION_CATEGORICAL = [
    "contract_type",
    "internet_service",
    "payment_method",
]

RANDOM_STATE = 42


def build_preprocessor(
    features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric_features = [f for f in features if f not in categorical_features]
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def _validate_columns(data: pd.DataFrame, features: list[str], target: str) -> None:
    missing = [c for c in features + [target] if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _split(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data[features]
    y = data[target]
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y if stratify else None,
    )


def train_regression(
    data: pd.DataFrame,
    output_dir: Path = ARTIFACTS_DIR,
) -> dict:
    features = REGRESSION_FEATURES
    _validate_columns(data, features, REGRESSION_TARGET)

    X_train, X_test, y_train, y_test = _split(
        data, features, REGRESSION_TARGET
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features, REGRESSION_CATEGORICAL)),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "target_mean": float(y_train.mean()),
    }

    return _save_artifact(
        pipeline, "regression", REGRESSION_TARGET, features, metrics, output_dir
    )


def train_classification(
    data: pd.DataFrame,
    output_dir: Path = ARTIFACTS_DIR,
) -> dict:
    features = CLASSIFICATION_FEATURES
    _validate_columns(data, features, CLASSIFICATION_TARGET)

    X_train, X_test, y_train, y_test = _split(
        data, features, CLASSIFICATION_TARGET, stratify=True
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features, CLASSIFICATION_CATEGORICAL)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    early_stopping=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "positive_rate": float(y_train.mean()),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    return _save_artifact(
        pipeline,
        "classification",
        CLASSIFICATION_TARGET,
        features,
        metrics,
        output_dir,
    )


def _save_artifact(
    pipeline: Pipeline,
    task: str,
    target: str,
    features: list[str],
    metrics: dict,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": pipeline,
        "task": task,
        "target": target,
        "features": features,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    model_path = output_dir / f"{task}_model.joblib"
    joblib.dump(artifact, model_path)
    return {"path": str(model_path), **artifact}
