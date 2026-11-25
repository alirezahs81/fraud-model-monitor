"""Append-only prediction log (JSON Lines)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import FEATURE_COLS, PREDICTION_LOG


def append_prediction(
    features: dict[str, float],
    probability: float,
    prediction: int,
    label: int | None = None,
    path: Path = PREDICTION_LOG,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "probability": float(probability),
        "prediction": int(prediction),
        "label": label,
        "features": {k: float(features[k]) for k in FEATURE_COLS if k in features},
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def load_predictions(path: Path = PREDICTION_LOG) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Expand features for drift analysis
    if "features" in df.columns:
        feat_df = pd.json_normalize(df["features"])
        df = pd.concat([df.drop(columns=["features"]), feat_df], axis=1)
    return df


def clear_log(path: Path = PREDICTION_LOG) -> None:
    path = Path(path)
    if path.exists():
        path.write_text("", encoding="utf-8")
