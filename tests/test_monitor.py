from src.data_synth import generate_transactions
from src.logging_store import clear_log
from src.model_ops import train_baseline_model
from src.monitor import FraudMonitor


def setup_module():
    clear_log()
    train_baseline_model()


def test_predict_one_structure():
    clear_log()
    mon = FraudMonitor()
    row = generate_transactions(n=1, random_state=9).iloc[0]
    feats = {c: float(row[c]) for c in row.index if c != "Class"}
    out = mon.predict_one(feats, label=int(row["Class"]))
    assert out["prediction"] in (0, 1)
    assert 0.0 <= out["probability"] <= 1.0


def test_status_after_batch():
    clear_log()
    mon = FraudMonitor()
    df = generate_transactions(n=200, random_state=3)
    mon.predict_batch(df, log=True)
    status = mon.status()
    assert status["n_predictions"] == 200
    assert status["status"] in {"ok", "warn", "alert"}
    assert status["performance"]["n_labeled"] == 200
