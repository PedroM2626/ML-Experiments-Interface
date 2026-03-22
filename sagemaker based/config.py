"""
AutoML Studio - Global Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MLRUNS_DIR = BASE_DIR / "mlruns"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
MLRUNS_DIR.mkdir(exist_ok=True)

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:///{MLRUNS_DIR.as_posix()}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "AutoML-Studio")

# ── App ────────────────────────────────────────────────────────────────────────
APP_NAME = "AutoML Studio"
APP_ICON = "🚀"
APP_VERSION = "1.0.0"

# ── AutoML ─────────────────────────────────────────────────────────────────────
QUICK_BUILD_TIME = 120        # seconds
STANDARD_BUILD_TIME = None    # no limit

SUPPORTED_PROBLEM_TYPES = ["binary", "multiclass", "regression"]

PROBLEM_TYPE_LABELS = {
    "binary": "Binary Classification",
    "multiclass": "Multi-class Classification",
    "regression": "Regression",
}

PROBLEM_TYPE_METRICS = {
    "binary": "roc_auc",
    "multiclass": "accuracy",
    "regression": "rmse",
}

# ── File Upload ────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = ["csv", "xlsx", "xls", "parquet"]
MAX_UPLOAD_SIZE_MB = 200

# ── UI Colors (matching SageMaker Canvas style) ────────────────────────────────
COLORS = {
    "primary": "#7C3AED",        # Purple-600
    "primary_light": "#A78BFA",  # Purple-400
    "secondary": "#2563EB",      # Blue-600
    "success": "#059669",        # Emerald-600
    "warning": "#D97706",        # Amber-600
    "danger": "#DC2626",         # Red-600
    "dark_bg": "#0F172A",        # Slate-900
    "card_bg": "#1E293B",        # Slate-800
    "border": "#334155",         # Slate-700
    "text_primary": "#F1F5F9",   # Slate-100
    "text_muted": "#94A3B8",     # Slate-400
}
