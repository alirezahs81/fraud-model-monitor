"""Performance metrics when ground-truth labels arrive later."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def labeled_performance(df: pd.DataFrame) -> dict[str, Any]:
    """Compute metrics on log rows that have a non-null label."""
    if df.empty or "label" not in df.columns:
        return {"n_labeled": 0, "metrics": None}

    labeled = df.dropna(subset=["label"]).copy()
    if labeled.empty:
        return {"n_labeled": 0, "metrics": None}

    y_true = labeled["label"].astype(int).to_numpy()
    y_pred = labeled["prediction"].astype(int).to_numpy()
    y_prob = labeled["probability"].astype(float).to_numpy()

    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    else:
        metrics["roc_auc"] = None

    return {"n_labeled": int(len(labeled)), "metrics": metrics}
