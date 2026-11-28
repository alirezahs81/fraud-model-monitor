from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOG_DIR = DATA_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
EXTERNAL_DIR = MODELS_DIR / "external"

for _d in (LOG_DIR, MODELS_DIR, EXTERNAL_DIR, DATA_DIR / "reference"):
    _d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "fraud_monitor_model.joblib"
REFERENCE_PATH = DATA_DIR / "reference" / "baseline_stats.joblib"
PREDICTION_LOG = LOG_DIR / "predictions.jsonl"

# Optional: path to a trained pipeline from bank-fraud-detection
# (models/best_fraud_model.joblib). Overrides the synthetic baseline when set.
EXTERNAL_MODEL_PATH = os.environ.get("FRAUD_MODEL_PATH", "").strip()

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL = "Class"

# Drift thresholds (PSI)
PSI_WARN = 0.1
PSI_ALERT = 0.25

# Score distribution drift (KS)
KS_WARN = 0.1
KS_ALERT = 0.2

RANDOM_STATE = 42
N_FEATURES = 29
