"""Train baseline, score matched + drifted windows, print monitor status."""

from __future__ import annotations

import json

from src.data_synth import generate_transactions
from src.logging_store import clear_log
from src.model_ops import train_baseline_model
from src.monitor import FraudMonitor


def main() -> None:
    clear_log()
    info = train_baseline_model()
    print("Baseline model:", json.dumps(info, indent=2))

    mon = FraudMonitor()

    # In-distribution window
    matched = generate_transactions(n=400, fraud_rate=0.02, shift=0.0, random_state=1)
    mon.predict_batch(matched, log=True)
    print("\nStatus after matched traffic:")
    print(json.dumps(mon.status(), indent=2, default=str))

    # Drifted window (feature mean shift + amount scale)
    drifted = generate_transactions(
        n=400, fraud_rate=0.02, shift=1.5, amount_scale=3.0, random_state=2
    )
    mon.predict_batch(drifted, log=True)
    print("\nStatus after drifted traffic:")
    print(json.dumps(mon.status(), indent=2, default=str))


if __name__ == "__main__":
    main()
