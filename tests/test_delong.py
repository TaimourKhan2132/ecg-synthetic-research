"""Tests for the fast DeLong implementation (src/utils/delong.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "utils"))
from delong import delong_roc_test  # noqa: E402

CLASSES = ["NORM", "MI", "AFIB", "TACHY"]


def _toy(seed=0, n=600):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    strong = y * 1.2 + rng.normal(0, 1, n)
    weak = y * 0.3 + rng.normal(0, 1, n)
    return y, strong, weak


def test_aucs_match_sklearn():
    y, strong, weak = _toy()
    a1, a2, _ = delong_roc_test(y, strong, weak)
    assert a1 == pytest.approx(roc_auc_score(y, strong))
    assert a2 == pytest.approx(roc_auc_score(y, weak))


def test_aucs_match_sklearn_with_ties():
    y, strong, weak = _toy(seed=1)
    strong, weak = np.round(strong, 1), np.round(weak, 1)  # heavy ties exercise the midrank
    a1, a2, _ = delong_roc_test(y, strong, weak)
    assert a1 == pytest.approx(roc_auc_score(y, strong))
    assert a2 == pytest.approx(roc_auc_score(y, weak))


def test_clearly_better_classifier_is_significant():
    y, strong, weak = _toy(seed=2)
    _, _, p = delong_roc_test(y, strong, weak)
    assert p < 0.001


def test_identical_classifiers_are_not_significant():
    y, strong, _ = _toy(seed=3)
    a1, a2, p = delong_roc_test(y, strong, strong)
    assert a1 == a2
    assert p == pytest.approx(1.0)


def test_symmetric_in_argument_order():
    y, strong, weak = _toy(seed=4)
    _, _, p_ab = delong_roc_test(y, strong, weak)
    _, _, p_ba = delong_roc_test(y, weak, strong)
    assert p_ab == pytest.approx(p_ba)


def test_published_delong_table_reproduces():
    """Recompute the paper's DeLong table from the committed per-sample probabilities."""
    d = ROOT / "outputs" / "results" / "delong"
    if not (d / "probs_A.csv").exists():
        pytest.skip("committed probability files not available")
    probs = {k: pd.read_csv(d / f"probs_{k}.csv", index_col="filepath") for k in "ABC"}
    common = sorted(set(probs["A"].index) & set(probs["B"].index) & set(probs["C"].index))
    probs = {k: v.loc[common] for k, v in probs.items()}
    y = probs["A"]["true"].to_numpy()
    published = pd.read_csv(d / "delong_auc_test.csv").set_index("cls")
    for i, cls in enumerate(CLASSES):
        yk = (y == i).astype(int)
        col = f"p_{cls}"
        a_a, a_b, p_ba = delong_roc_test(yk, probs["A"][col].to_numpy(), probs["B"][col].to_numpy())
        _, a_c, p_ca = delong_roc_test(yk, probs["A"][col].to_numpy(), probs["C"][col].to_numpy())
        row = published.loc[cls]
        assert (a_a, a_b, a_c) == pytest.approx((row.auc_A, row.auc_B, row.auc_C), abs=1e-9)
        assert (p_ba, p_ca) == pytest.approx((row.p_BvsA, row.p_CvsA), rel=1e-6, abs=1e-12)
