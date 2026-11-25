import numpy as np
import pandas as pd

from src.config import FEATURE_COLS
from src.data_synth import generate_transactions
from src.drift import feature_psi_report, ks_score_drift, psi_for_feature


def test_psi_zero_on_same_distribution():
    x = np.random.default_rng(0).normal(size=500)
    assert psi_for_feature(x, x) < 0.05


def test_psi_increases_with_shift():
    ref = generate_transactions(n=800, shift=0.0, random_state=0)
    cur = generate_transactions(n=800, shift=2.0, random_state=1)
    report = feature_psi_report(ref[FEATURE_COLS], cur[FEATURE_COLS])
    assert report["max_psi"] > 0.1


def test_ks_detects_score_shift():
    a = np.random.default_rng(0).uniform(0, 0.2, size=300)
    b = np.random.default_rng(1).uniform(0.5, 1.0, size=300)
    out = ks_score_drift(a, b)
    assert out["ks"] > 0.3
