# Fraud Model Monitor (MLOps)

[![CI](https://github.com/alirezahs81/fraud-model-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/fraud-model-monitor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Production-shaped monitoring** for credit-card fraud models — the layer that sits *after* training.

Complements [bank-fraud-detection](https://github.com/alirezahs81/bank-fraud-detection):

| Concern | What this repo does |
|---------|---------------------|
| Inference logging | JSONL store of features, scores, predictions, optional labels |
| **Data drift** | Population Stability Index (PSI) per feature vs training baseline |
| **Score drift** | Kolmogorov–Smirnov on prediction probabilities |
| Delayed labels | Accuracy / precision / recall / F1 / ROC-AUC when labels arrive |
| Status API | Aggregate `ok` / `warn` / `alert` for ops dashboards |

Uses a **synthetic** V1–V28 + Amount generator so CI stays offline (no Kaggle dependency). Swap in a real pipeline artifact when integrating with `bank-fraud-detection`.

---

## Architecture

```text
live transaction
      │
      ▼
  /predict  →  model score  →  JSONL log
                                  │
                                  ▼
                         /monitor/status
                    PSI · KS · labeled metrics
```

---

## Quick start

```bash
git clone https://github.com/alirezahs81/fraud-model-monitor.git
cd fraud-model-monitor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py                 # train baseline, score matched + drifted windows
pytest tests/ -v
uvicorn src.api:app --reload --port 8000
```

```bash
# health
curl -s http://localhost:8000/health

# monitor snapshot
curl -s http://localhost:8000/monitor/status
```

### Docker

```bash
docker build -t fraud-monitor .
docker run --rm -p 8000:8000 fraud-monitor
```

---

## Drift thresholds (defaults)

| Signal | Warn | Alert |
|--------|------|-------|
| Feature PSI (max) | ≥ 0.10 | ≥ 0.25 |
| Score KS | ≥ 0.10 | ≥ 0.20 |

Tune in `src/config.py`.

---

## Layout

```text
fraud-model-monitor/
├── src/
│   ├── data_synth.py      # synthetic PCA-like transactions
│   ├── model_ops.py       # train/load baseline + reference stats
│   ├── logging_store.py   # JSONL prediction log
│   ├── drift.py           # PSI + KS
│   ├── performance.py     # delayed-label metrics
│   ├── monitor.py         # orchestration
│   └── api.py             # FastAPI
├── tests/
├── main.py
└── Dockerfile
```

---

## Link to bank-fraud-detection

1. Train your production pipeline in `bank-fraud-detection` and export `models/best_fraud_model.joblib`.
2. Point `MODEL_PATH` at that artifact (same feature order: V1–V28, Amount).
3. Rebuild reference stats from the **training** frame only (`means`, `stds`, score sample).
4. Route live traffic through `/predict` and poll `/monitor/status`.

---

## License

[MIT](LICENSE) © 2026 Alireza Hasheminasab
