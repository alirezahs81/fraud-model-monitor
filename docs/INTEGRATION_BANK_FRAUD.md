# Integrating `bank-fraud-detection` with this monitor

Both repos share the same feature contract: **V1–V28 + Amount**, label **Class**.

```text
bank-fraud-detection                 fraud-model-monitor
────────────────────                 ───────────────────
train → models/best_fraud_model.joblib
              │
              ▼
     python -m src.register_model --model <path>
              │
              ▼
     /predict  →  JSONL log  →  /monitor/status (PSI / KS / metrics)
```

## Steps

### 1. Train the production pipeline

```bash
cd bank-fraud-detection
# dataset + deps as in that README
python -m src.main
# → models/best_fraud_model.joblib
```

### 2. Register the artifact in the monitor

```bash
cd fraud-model-monitor
pip install -r requirements.txt

python -m src.register_model \
  --model /absolute/path/to/bank-fraud-detection/models/best_fraud_model.joblib
```

Optional — better drift baseline from the **training** frame only:

```bash
python -m src.register_model \
  --model .../best_fraud_model.joblib \
  --reference-csv /path/to/train_features.csv
```

`train_features.csv` must include columns `V1`…`V28`, `Amount` (and optionally `Class`).

### 3. Run the API

```bash
uvicorn src.api:app --reload --port 8000
curl -s http://localhost:8000/health
curl -s http://localhost:8000/monitor/status
```

### Alternative: environment variable

Without copying via `register_model`, point the monitor at the artifact at process start:

```bash
export FRAUD_MODEL_PATH=/path/to/best_fraud_model.joblib
# Still need reference stats — prefer register_model once, or train_baseline_model for synthetic ref only
uvicorn src.api:app --port 8000
```

`register_model` is preferred: it validates `predict_proba`, stores a copy under `models/external/`, and writes reference stats.

## Notes

- CI in this repo keeps using the **synthetic** baseline so GitHub Actions stays offline (no Kaggle).
- If the bank-fraud artifact is a full sklearn/imblearn **Pipeline**, pass a DataFrame with the raw feature columns (same as `FraudPredictor` in that project).
- Do not fit drift reference on the live log; use training (or a held-out reference window) only.
