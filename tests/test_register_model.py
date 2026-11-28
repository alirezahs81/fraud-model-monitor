from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLS, MODEL_PATH, REFERENCE_PATH
from src.data_synth import generate_transactions
from src.monitor import FraudMonitor
from src.register_model import register_model
from src.logging_store import clear_log


def test_register_sklearn_pipeline(tmp_path: Path):
    df = generate_transactions(n=400, random_state=11)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    pipe.fit(df[FEATURE_COLS], df["Class"])
    artifact = tmp_path / "best_fraud_model.joblib"
    joblib.dump(pipe, artifact)

    dest_model = tmp_path / "monitor_model.joblib"
    dest_ref = tmp_path / "ref.joblib"
    info = register_model(
        artifact,
        dest_model=dest_model,
        dest_reference=dest_ref,
    )
    assert Path(info["model_path"]).exists()
    assert Path(info["reference_path"]).exists()

    clear_log()
    mon = FraudMonitor(model_path=dest_model, reference_path=dest_ref)
    row = {c: float(df.iloc[0][c]) for c in FEATURE_COLS}
    out = mon.predict_one(row, label=int(df.iloc[0]["Class"]))
    assert out["prediction"] in (0, 1)
    status = mon.status()
    assert status["n_predictions"] >= 1
