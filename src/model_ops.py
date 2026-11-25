"""Train a leakage-safe sklearn pipeline and persist baseline feature stats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLS, MODEL_PATH, REFERENCE_PATH, TARGET_COL
from src.data_synth import generate_transactions


def train_baseline_model(
    df: pd.DataFrame | None = None,
    model_path: Path = MODEL_PATH,
    reference_path: Path = REFERENCE_PATH,
) -> dict[str, Any]:
    df = df if df is not None else generate_transactions(n=3000, fraud_rate=0.02)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(X, y)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)

    # Reference stats for drift (train distribution)
    ref = {
        "feature_cols": FEATURE_COLS,
        "means": X.mean().to_dict(),
        "stds": X.std(ddof=0).replace(0, 1.0).to_dict(),
        "quantiles": {
            c: np.quantile(X[c].to_numpy(), [0, 0.25, 0.5, 0.75, 1.0]).tolist()
            for c in FEATURE_COLS
        },
        "score_ref": pipe.predict_proba(X)[:, 1].tolist()[:5000],
    }
    reference_path = Path(reference_path)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ref, reference_path)

    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = float((pred == y.to_numpy()).mean())
    return {
        "model_path": str(model_path),
        "reference_path": str(reference_path),
        "n_train": len(df),
        "train_accuracy": round(acc, 4),
        "fraud_rate": float(y.mean()),
    }


def load_model(path: Path = MODEL_PATH):
    path = Path(path)
    if not path.exists():
        train_baseline_model(model_path=path)
    return joblib.load(path)


def load_reference(path: Path = REFERENCE_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        train_baseline_model(reference_path=path)
    return joblib.load(path)
