"""Synthetic PCA-style transactions for offline demo and CI.

Mirrors the shape of the public credit-card fraud dataset (V1–V28 + Amount + Class)
without requiring Kaggle downloads. Not real card data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FEATURE_COLS, RANDOM_STATE, TARGET_COL


def generate_transactions(
    n: int = 2000,
    fraud_rate: float = 0.02,
    shift: float = 0.0,
    amount_scale: float = 1.0,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a table of synthetic transactions.

    ``shift`` moves the mean of PCA-like features (simulates covariate drift).
    ``amount_scale`` multiplies Amount (simulates scale drift).
    """
    rng = np.random.default_rng(random_state)
    n_fraud = max(1, int(n * fraud_rate))
    n_legit = n - n_fraud

    def _block(size: int, fraud: bool) -> pd.DataFrame:
        # Fraud rows get a slightly different mean so a linear model can separate them
        mean = (1.2 if fraud else 0.0) + shift
        feats = rng.normal(loc=mean, scale=1.0, size=(size, 28))
        amount = rng.lognormal(mean=3.5 if not fraud else 4.2, sigma=0.8, size=size)
        amount = amount * amount_scale
        data = {f"V{i+1}": feats[:, i] for i in range(28)}
        data["Amount"] = amount
        data[TARGET_COL] = 1 if fraud else 0
        return pd.DataFrame(data)

    df = pd.concat(
        [_block(n_legit, False), _block(n_fraud, True)],
        ignore_index=True,
    )
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)[FEATURE_COLS + [TARGET_COL]]
