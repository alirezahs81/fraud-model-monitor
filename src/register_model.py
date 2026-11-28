"""Register an external sklearn pipeline (e.g. bank-fraud-detection) for monitoring.

Expected contract:
- joblib artifact with ``predict_proba``
- features in order V1..V28, Amount (or a DataFrame with those columns)

Usage::

    python -m src.register_model --model /path/to/best_fraud_model.joblib

Optional reference CSV with the same feature columns (training frame only)::

    python -m src.register_model --model ... --reference-csv train_features.csv
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    EXTERNAL_DIR,
    FEATURE_COLS,
    MODEL_PATH,
    REFERENCE_PATH,
    TARGET_COL,
)
from src.data_synth import generate_transactions


def _validate_estimator(model: Any) -> None:
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Artifact must expose predict_proba (sklearn classifier / pipeline)."
        )


def _score_matrix(model: Any, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X[FEATURE_COLS])
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("predict_proba must return shape (n, n_classes) with class 1 = fraud")
    return proba[:, 1]


def build_reference_stats(model: Any, frame: pd.DataFrame) -> dict[str, Any]:
    X = frame[FEATURE_COLS]
    scores = _score_matrix(model, X)
    return {
        "feature_cols": FEATURE_COLS,
        "means": X.mean().to_dict(),
        "stds": X.std(ddof=0).replace(0, 1.0).to_dict(),
        "quantiles": {
            c: np.quantile(X[c].to_numpy(), [0, 0.25, 0.5, 0.75, 1.0]).tolist()
            for c in FEATURE_COLS
        },
        "score_ref": scores[:5000].astype(float).tolist(),
        "source": "external",
    }


def register_model(
    model_path: Path,
    reference_csv: Path | None = None,
    dest_model: Path = MODEL_PATH,
    dest_reference: Path = REFERENCE_PATH,
) -> dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    _validate_estimator(model)

    if reference_csv is not None:
        frame = pd.read_csv(reference_csv)
        missing = [c for c in FEATURE_COLS if c not in frame.columns]
        if missing:
            raise ValueError(f"Reference CSV missing columns: {missing}")
    else:
        # Offline-safe baseline window for drift (not a substitute for real train data)
        frame = generate_transactions(n=3000, fraud_rate=0.02)

    # Smoke-score a few rows to fail fast on feature mismatch
    _ = _score_matrix(model, frame.head(5))

    dest_model = Path(dest_model)
    dest_model.parent.mkdir(parents=True, exist_ok=True)
    # Keep a copy under models/external for auditability
    external_copy = EXTERNAL_DIR / model_path.name
    shutil.copy2(model_path, external_copy)
    joblib.dump(model, dest_model)

    ref = build_reference_stats(model, frame)
    ref["external_copy"] = str(external_copy)
    ref["registered_from"] = str(model_path.resolve())
    dest_reference = Path(dest_reference)
    dest_reference.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ref, dest_reference)

    return {
        "model_path": str(dest_model),
        "reference_path": str(dest_reference),
        "external_copy": str(external_copy),
        "n_reference_rows": len(frame),
        "has_labels": TARGET_COL in frame.columns,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to bank-fraud best_fraud_model.joblib (or any sklearn pipeline)",
    )
    parser.add_argument(
        "--reference-csv",
        default=None,
        help="Optional training-frame CSV with V1..V28, Amount for drift baseline",
    )
    args = parser.parse_args(argv)
    info = register_model(
        Path(args.model),
        reference_csv=Path(args.reference_csv) if args.reference_csv else None,
    )
    print("Registered external model for monitoring:")
    for k, v in info.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
