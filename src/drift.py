"""Data and score drift metrics.

Population Stability Index (PSI) for features and Kolmogorov–Smirnov on
prediction scores. Thresholds come from config (warn / alert).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats  # may not be available — fallback below

from src.config import FEATURE_COLS, KS_ALERT, KS_WARN, PSI_ALERT, PSI_WARN


def _psi_from_counts(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    expected = expected.astype(float)
    actual = actual.astype(float)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    expected = np.clip(expected, eps, None)
    actual = np.clip(actual, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_for_feature(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    # Shared edges from reference quantiles
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        # Degenerate feature
        return 0.0
    exp_counts, _ = np.histogram(ref, bins=edges)
    act_counts, _ = np.histogram(cur, bins=edges)
    return _psi_from_counts(exp_counts, act_counts)


def feature_psi_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
) -> dict[str, Any]:
    per_feature = {}
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        val = psi_for_feature(reference_df[col].to_numpy(), current_df[col].to_numpy())
        level = "ok"
        if val >= PSI_ALERT:
            level = "alert"
        elif val >= PSI_WARN:
            level = "warn"
        per_feature[col] = {"psi": round(val, 4), "level": level}

    values = [v["psi"] for v in per_feature.values()] or [0.0]
    max_psi = max(values)
    mean_psi = float(np.mean(values))
    overall = "ok"
    if max_psi >= PSI_ALERT:
        overall = "alert"
    elif max_psi >= PSI_WARN:
        overall = "warn"

    return {
        "overall": overall,
        "max_psi": round(max_psi, 4),
        "mean_psi": round(mean_psi, 4),
        "features": per_feature,
    }


def ks_score_drift(ref_scores: np.ndarray, cur_scores: np.ndarray) -> dict[str, Any]:
    ref_scores = np.asarray(ref_scores, dtype=float)
    cur_scores = np.asarray(cur_scores, dtype=float)
    if len(ref_scores) < 5 or len(cur_scores) < 5:
        return {"ks": 0.0, "level": "ok", "note": "insufficient samples"}
    try:
        ks = float(stats.ks_2samp(ref_scores, cur_scores).statistic)
    except Exception:
        # Pure numpy fallback: max CDF gap on pooled grid
        grid = np.sort(np.concatenate([ref_scores, cur_scores]))
        ref_cdf = np.searchsorted(np.sort(ref_scores), grid, side="right") / len(ref_scores)
        cur_cdf = np.searchsorted(np.sort(cur_scores), grid, side="right") / len(cur_scores)
        ks = float(np.max(np.abs(ref_cdf - cur_cdf)))

    level = "ok"
    if ks >= KS_ALERT:
        level = "alert"
    elif ks >= KS_WARN:
        level = "warn"
    return {"ks": round(ks, 4), "level": level}
