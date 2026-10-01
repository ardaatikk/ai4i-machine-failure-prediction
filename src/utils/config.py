from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset configuration
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)

DATASET_FILENAME = "ai4i2020.csv"
DATASET_ZIP_FILENAME = "ai4i2020.zip"

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / DATASET_FILENAME

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"

BASELINE_MODEL_PATH = (
    MODELS_DIR / "baseline_logistic_regression.joblib"
)

ENGINEERED_MODEL_PATH = (
    MODELS_DIR / "engineered_logistic_regression.joblib"
)

RANDOM_FOREST_MODEL_PATH = (
    MODELS_DIR / "random_forest.joblib"
)

GRADIENT_BOOSTING_MODEL_PATH = (
    MODELS_DIR / "gradient_boosting.joblib"
)

FINAL_MODEL_PATH = RANDOM_FOREST_MODEL_PATH

# Report paths
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FINAL_EVALUATION_REPORT_PATH = (
    REPORTS_DIR / "final_evaluation.json"
)