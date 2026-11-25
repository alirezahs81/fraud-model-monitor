"""FastAPI surface for fraud scoring + monitoring."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import FEATURE_COLS
from src.monitor import FraudMonitor

app = FastAPI(
    title="Fraud Model Monitor",
    description=(
        "MLOps monitor for credit-card fraud models: score transactions, "
        "log predictions, detect PSI/KS drift, and report delayed-label metrics."
    ),
    version="1.0.0",
)

_monitor: FraudMonitor | None = None


def get_monitor() -> FraudMonitor:
    global _monitor
    if _monitor is None:
        try:
            _monitor = FraudMonitor()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _monitor


class TransactionIn(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Map of feature name → value (V1–V28, Amount)",
    )
    label: int | None = Field(default=None, description="Optional delayed ground truth")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        get_monitor()
        return {"status": "ok", "model_ready": True}
    except HTTPException:
        return {"status": "degraded", "model_ready": False}


@app.post("/predict")
def predict(body: TransactionIn) -> dict[str, Any]:
    missing = [c for c in FEATURE_COLS if c not in body.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}",
        )
    mon = get_monitor()
    return mon.predict_one(body.features, label=body.label, threshold=body.threshold)


@app.get("/monitor/status")
def monitor_status() -> dict[str, Any]:
    return get_monitor().status()
