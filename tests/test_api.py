from fastapi.testclient import TestClient

from src.api import app
from src.config import FEATURE_COLS
from src.data_synth import generate_transactions
from src.logging_store import clear_log
from src.model_ops import train_baseline_model


def setup_module():
    clear_log()
    train_baseline_model()


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_ready"] is True


def test_predict_and_status():
    clear_log()
    client = TestClient(app)
    row = generate_transactions(n=1, random_state=5).iloc[0]
    features = {c: float(row[c]) for c in FEATURE_COLS}
    r = client.post(
        "/predict",
        json={"features": features, "label": int(row["Class"])},
    )
    assert r.status_code == 200
    body = r.json()
    assert "probability" in body

    s = client.get("/monitor/status")
    assert s.status_code == 200
    assert s.json()["n_predictions"] >= 1
