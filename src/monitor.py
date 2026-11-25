"""High-level monitor: score transactions, log, and report health."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import FEATURE_COLS, MODEL_PATH, REFERENCE_PATH
from src.drift import feature_psi_report, ks_score_drift
from src.logging_store import append_prediction, load_predictions
from src.model_ops import load_model, load_reference
from src.performance import labeled_performance


class FraudMonitor:
    def __init__(self, model_path=MODEL_PATH, reference_path=REFERENCE_PATH):
        self.model = load_model(model_path)
        self.reference = load_reference(reference_path)

    def predict_one(
        self,
        features: dict[str, float],
        label: int | None = None,
        threshold: float = 0.5,
        log: bool = True,
    ) -> dict[str, Any]:
        row = {c: float(features.get(c, 0.0)) for c in FEATURE_COLS}
        X = pd.DataFrame([row])[FEATURE_COLS]
        proba = float(self.model.predict_proba(X)[0, 1])
        pred = int(proba >= threshold)
        if log:
            append_prediction(row, proba, pred, label=label)
        return {
            "prediction": pred,
            "label_name": "fraud" if pred == 1 else "legitimate",
            "probability": round(proba, 6),
            "threshold": threshold,
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
        log: bool = True,
    ) -> pd.DataFrame:
        X = df[FEATURE_COLS]
        proba = self.model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
        out = df.copy()
        out["probability"] = proba
        out["prediction"] = pred
        if log:
            labels = df["Class"].tolist() if "Class" in df.columns else [None] * len(df)
            for i, row in enumerate(X.to_dict(orient="records")):
                append_prediction(row, float(proba[i]), int(pred[i]), label=labels[i])
        return out

    def status(self) -> dict[str, Any]:
        logs = load_predictions()
        ref_scores = np.asarray(self.reference.get("score_ref", []), dtype=float)

        if logs.empty:
            return {
                "status": "ok",
                "n_predictions": 0,
                "drift": None,
                "score_drift": None,
                "performance": {"n_labeled": 0, "metrics": None},
                "message": "No live predictions logged yet.",
            }

        # Feature drift vs training reference means/std → rebuild ref frame from quantiles
        ref_means = self.reference["means"]
        # Synthetic reference sample from stored quantiles mid-points is heavy;
        # use logged features vs a reconstructed normal from mean/std for PSI edges:
        rng = np.random.default_rng(0)
        ref_frame = {}
        for c in FEATURE_COLS:
            mu = float(ref_means[c])
            sd = float(self.reference["stds"].get(c, 1.0)) or 1.0
            ref_frame[c] = rng.normal(mu, sd, size=min(2000, max(len(logs) * 2, 200)))
        ref_df = pd.DataFrame(ref_frame)

        cur_cols = [c for c in FEATURE_COLS if c in logs.columns]
        drift = feature_psi_report(ref_df[cur_cols], logs[cur_cols], feature_cols=cur_cols)

        cur_scores = logs["probability"].to_numpy()
        score_drift = ks_score_drift(ref_scores, cur_scores) if len(ref_scores) else {
            "ks": 0.0,
            "level": "ok",
        }

        perf = labeled_performance(logs)

        levels = {drift["overall"], score_drift["level"]}
        if "alert" in levels:
            overall = "alert"
        elif "warn" in levels:
            overall = "warn"
        else:
            overall = "ok"

        return {
            "status": overall,
            "n_predictions": int(len(logs)),
            "drift": {
                "overall": drift["overall"],
                "max_psi": drift["max_psi"],
                "mean_psi": drift["mean_psi"],
                "top_features": sorted(
                    (
                        {"feature": k, **v}
                        for k, v in drift["features"].items()
                    ),
                    key=lambda x: x["psi"],
                    reverse=True,
                )[:5],
            },
            "score_drift": score_drift,
            "performance": perf,
        }
