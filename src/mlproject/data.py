from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlproject.config import DATA_DIR

REGRESSION_TARGET = "energy_kwh"
CLASSIFICATION_TARGET = "churned"

PROPERTY_TYPES = np.array(["apartment", "house", "office"])
CONTRACT_TYPES = np.array(["month-to-month", "one_year", "two_year"])
INTERNET_SERVICES = np.array(["fiber", "dsl", "none"])
PAYMENT_METHODS = np.array(["autopay", "credit_card", "mailed_check"])


def generate_regression_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    size_sqm = rng.uniform(35, 320, n_samples)
    occupants = rng.integers(1, 7, n_samples)
    insulation_rating = rng.integers(1, 6, n_samples)
    avg_temp_c = rng.uniform(2, 38, n_samples)
    building_age = rng.integers(0, 70, n_samples)
    has_solar = rng.binomial(1, 0.35, n_samples)
    property_type = rng.choice(PROPERTY_TYPES, n_samples, p=[0.5, 0.35, 0.15])

    type_effect = pd.Series(property_type).map(
        {"apartment": 400.0, "house": 950.0, "office": 1800.0}
    ).to_numpy()

    energy = (
        9.5 * size_sqm
        + 210.0 * occupants
        - 420.0 * insulation_rating
        + 46.0 * avg_temp_c
        + 18.0 * building_age
        - 1700.0 * has_solar
        + 0.045 * size_sqm * occupants
        + 0.09 * size_sqm * avg_temp_c
        + type_effect
        + rng.normal(0, 480, n_samples)
    )
    energy = np.clip(energy, 250, None)

    return pd.DataFrame(
        {
            "size_sqm": np.round(size_sqm, 2),
            "occupants": occupants,
            "insulation_rating": insulation_rating,
            "avg_temp_c": np.round(avg_temp_c, 2),
            "building_age_years": building_age,
            "has_solar": has_solar,
            "property_type": property_type,
            REGRESSION_TARGET: np.round(energy, 2),
        }
    )


def generate_classification_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(0, 73, n_samples)
    monthly_charges = rng.uniform(20, 150, n_samples)
    support_tickets = rng.poisson(1.6, n_samples)
    contract_type = rng.choice(CONTRACT_TYPES, n_samples, p=[0.55, 0.25, 0.20])
    internet_service = rng.choice(INTERNET_SERVICES, n_samples, p=[0.5, 0.35, 0.15])
    payment_method = rng.choice(
        PAYMENT_METHODS, n_samples, p=[0.45, 0.35, 0.20]
    )
    satisfaction = rng.integers(1, 11, n_samples)
    has_partner = rng.binomial(1, 0.5, n_samples)

    contract_effect = pd.Series(contract_type).map(
        {"month-to-month": 1.05, "one_year": -0.35, "two_year": -1.15}
    ).to_numpy()
    internet_effect = pd.Series(internet_service).map(
        {"fiber": 0.55, "dsl": 0.0, "none": -0.45}
    ).to_numpy()
    payment_effect = pd.Series(payment_method).map(
        {"autopay": -0.45, "credit_card": 0.0, "mailed_check": 0.55}
    ).to_numpy()

    logits = (
        -2.4
        - 0.035 * tenure_months
        + 0.018 * monthly_charges
        + 0.32 * support_tickets
        - 0.42 * satisfaction
        - 0.35 * has_partner
        + contract_effect
        + internet_effect
        + payment_effect
        + 0.004 * monthly_charges * support_tickets
        + rng.normal(0, 0.55, n_samples)
    )
    probability = 1.0 / (1.0 + np.exp(-logits))
    churned = rng.binomial(1, probability, n_samples)

    return pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": np.round(monthly_charges, 2),
            "support_tickets": support_tickets,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "satisfaction_score": satisfaction,
            "has_partner": has_partner,
            CLASSIFICATION_TARGET: churned,
        }
    )


def write_datasets(
    output_dir: Path = DATA_DIR,
    n_samples: int = 5000,
    seed: int = 42,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    regression_path = output_dir / "regression.csv"
    classification_path = output_dir / "classification.csv"

    generate_regression_data(n_samples, seed).to_csv(regression_path, index=False)
    generate_classification_data(n_samples, seed).to_csv(
        classification_path, index=False
    )

    return regression_path, classification_path
