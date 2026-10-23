# Tabular ML Pipeline

An end-to-end, reproducible machine learning project containing two models trained on
synthetic tabular data:

- **Regression** - predicts a building's annual energy consumption (`energy_kwh`).
- **Classification** - predicts customer churn (`churned`).

Both models are built with scikit-learn pipelines (preprocessing + gradient boosting),
evaluated on a held-out test split, and serialized to `joblib` artifacts for reuse.

## Results

Trained on 5,000 synthetic rows with an 80/20 train/test split.

| Task | Metric | Score |
| --- | --- | --- |
| Regression | MAE | 402.92 kWh |
| Regression | RMSE | 507.00 kWh |
| Regression | R2 | 0.908 |
| Classification | Accuracy | 0.843 |
| Classification | Precision | 0.433 |
| Classification | Recall | 0.705 |
| Classification | F1 | 0.537 |
| Classification | ROC-AUC | 0.878 |
| Classification | PR-AUC | 0.585 |

The classifier uses balanced class weights because churn is the minority class
(about 13% of customers).

## Project structure

```
.
|-- data/                       # generated datasets (gitignored)
|-- artifacts/                  # trained models + metrics (gitignored)
|-- examples/                   # sample input CSVs
|-- src/mlproject/
|   |-- config.py               # project paths
|   |-- data.py                 # synthetic data generation
|   |-- training.py             # preprocessing, training, evaluation
|   |-- predict.py              # inference helpers
|   |-- cli.py                  # train/predict command line interface
|   `-- __main__.py
|-- tests/
|-- pyproject.toml
`-- requirements.txt
```

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Train

Generate the synthetic datasets and train both models:

```powershell
python -m mlproject train
```

Train a single model or change the data size:

```powershell
python -m mlproject train --task regression
python -m mlproject train --task classification --n-samples 10000 --force
```

Outputs:

- `data/regression.csv`, `data/classification.csv`
- `artifacts/regression_model.joblib`, `artifacts/classification_model.joblib`
- `artifacts/regression_metrics.json`, `artifacts/classification_metrics.json`

## Predict

```powershell
python -m mlproject predict --task regression --input examples\regression_input.csv
python -m mlproject predict --task classification --input examples\classification_input.csv --threshold 0.5
python -m mlproject predict --task classification --input examples\classification_input.csv --output predictions.csv
```

Each artifact is a dictionary containing the fitted pipeline, feature list, target,
metrics, and training metadata. From Python:

```python
from mlproject.predict import predict_classification, predict_regression

row = {
    "size_sqm": 120.0,
    "occupants": 3,
    "insulation_rating": 4,
    "avg_temp_c": 22.5,
    "building_age_years": 12,
    "has_solar": 1,
    "property_type": "house",
}
print(predict_regression(row))

customer = {
    "tenure_months": 8,
    "monthly_charges": 89.5,
    "support_tickets": 4,
    "contract_type": "month-to-month",
    "internet_service": "fiber",
    "payment_method": "mailed_check",
    "satisfaction_score": 4,
    "has_partner": 0,
}
print(predict_classification(customer))
```

## Tests

```powershell
python -m pytest -q
```

Tests cover data generation, model quality thresholds, and end-to-end predictions.

## Data

The data is fully synthetic and seeded, so every run is reproducible.

- **Regression features:** property size, occupants, insulation rating, average
  temperature, building age, solar panels, property type.
- **Classification features:** tenure, monthly charges, support tickets, contract type,
  internet service, payment method, satisfaction score, partner status.

## Publish to GitHub

```powershell
git init
git add .
git commit -m "Add regression and classification ML pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
