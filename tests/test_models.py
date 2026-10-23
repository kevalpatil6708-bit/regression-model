from pathlib import Path

import pandas as pd
import pytest

from mlproject.data import (
    generate_classification_data,
    generate_regression_data,
)
from mlproject.predict import predict_classification, predict_regression
from mlproject.training import train_classification, train_regression


@pytest.fixture(scope="session")
def trained_models(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    output_dir = tmp_path_factory.mktemp("artifacts")

    regression_result = train_regression(
        generate_regression_data(800, seed=11), output_dir
    )
    classification_result = train_classification(
        generate_classification_data(800, seed=11), output_dir
    )

    return Path(regression_result["path"]), Path(classification_result["path"])


def test_regression_quality(trained_models: tuple[Path, Path]) -> None:
    model_file, _ = trained_models
    from mlproject.predict import load_artifact

    metrics = load_artifact(model_file)["metrics"]

    assert metrics["r2"] > 0.7
    assert metrics["mae"] > 0


def test_classification_quality(trained_models: tuple[Path, Path]) -> None:
    _, model_file = trained_models
    from mlproject.predict import load_artifact

    metrics = load_artifact(model_file)["metrics"]

    assert metrics["roc_auc"] > 0.7
    assert metrics["accuracy"] > 0.65
    assert len(metrics["confusion_matrix"]) == 2


def test_predictions(trained_models: tuple[Path, Path]) -> None:
    regression_file, classification_file = trained_models

    regression_input = pd.DataFrame(
        [
            {
                "size_sqm": 120.0,
                "occupants": 3,
                "insulation_rating": 4,
                "avg_temp_c": 22.5,
                "building_age_years": 12,
                "has_solar": 1,
                "property_type": "house",
            }
        ]
    )
    regression_output = predict_regression(regression_input, regression_file)

    assert "predicted_energy_kwh" in regression_output.columns
    assert regression_output["predicted_energy_kwh"].iloc[0] > 0

    classification_input = pd.DataFrame(
        [
            {
                "tenure_months": 8,
                "monthly_charges": 89.5,
                "support_tickets": 4,
                "contract_type": "month-to-month",
                "internet_service": "fiber",
                "payment_method": "mailed_check",
                "satisfaction_score": 4,
                "has_partner": 0,
            }
        ]
    )
    classification_output = predict_classification(
        classification_input, classification_file
    )

    probability = classification_output["churn_probability"].iloc[0]
    assert 0.0 <= probability <= 1.0
    assert classification_output["churn_prediction"].iloc[0] in {0, 1}
