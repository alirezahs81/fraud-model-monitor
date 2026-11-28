# Fraud Model Monitor (MLOps)

[![CI](https://github.com/alirezahs81/fraud-model-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/fraud-model-monitor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Production-shaped monitoring** for credit-card fraud models — the layer that sits *after* training.

Pairs with [**bank-fraud-detection**](https://github.com/alirezahs81/bank-fraud-detection):

| Concern | What this repo does |
|---------|---------------------|
| Inference logging | JSONL store of features, scores, predictions, optional labels |
| **Data drift** | Population Stability Index (PSI) per feature vs training baseline |
| **Score drift** | Kolmogorov–Smirnov on prediction probabilities |
| Delayed labels | Accuracy / precision / recall / F1 / ROC-AUC when labels arrive |
| Status API | Aggregate `ok` / `warn` / `alert` for ops dashboards |
| **External model** | `python -m src.register_model --model best_fraud_model.joblib` |

Default CI path uses a **synthetic** V1–V28 + Amount generator (no Kaggle). Production path loads the trained bank-fraud pipeline.

---

## Architecture

```text
bank-fraud-detection                 this repo
train → best_fraud_model.joblib  →  register_model
                                         │
live transaction ──► /predict ──► JSONL log ──► /monitor/status
                                              PSI · KS · labeled metrics
```

---

## Quick start (synthetic baseline)

```bash
git clone https://github.com/alirezahs81/fraud-model-monitor.git
cd fraud-model-monitor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py
pytest tests/ -v
uvicorn src.api:app --reload --port 8000
```

---

## Use the real bank-fraud model

```bash
# after training in bank-fraud-detection:
python -m src.register_model \
  --model /path/to/bank-fraud-detection/models/best_fraud_model.joblib

# optional: training-frame CSV for a true drift baseline
python -m src.register_model \
  --model .../best_fraud_model.joblib \
  --reference-csv /path/to/train_features.csv

uvicorn src.api:app --port 8000
```

Full walkthrough: [docs/INTEGRATION_BANK_FRAUD.md](docs/INTEGRATION_BANK_FRAUD.md)

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
│   ├── data_synth.py
│   ├── model_ops.py
│   ├── register_model.py   # bank-fraud artifact import
│   ├── logging_store.py
│   ├── drift.py
│   ├── performance.py
│   ├── monitor.py
│   └── api.py
├── docs/INTEGRATION_BANK_FRAUD.md
├── tests/
└── Dockerfile
```

---

## License

[MIT](LICENSE) © 2026 Alireza Hasheminasab
