from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REGRESSION_MODEL_FILE = ARTIFACTS_DIR / "regression_model.joblib"
CLASSIFICATION_MODEL_FILE = ARTIFACTS_DIR / "classification_model.joblib"
