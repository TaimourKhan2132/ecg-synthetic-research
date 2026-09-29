# =============================================================================
# delong.py — fast DeLong test for two correlated ROC-AUCs (Sun & Xu, 2014).
#
# Pure numpy/scipy so it can be unit-tested without torch. Used by
# delong_test.py, which recovers per-class probabilities from the checkpoints.
# =============================================================================
import numpy as np
import scipy.stats


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_T, m):
    n = preds_sorted_T.shape[1] - m
    pos = preds_sorted_T[:, :m]
    neg = preds_sorted_T[:, m:]
    k = preds_sorted_T.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r]); tz[r] = _midrank(preds_sorted_T[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, cov


def delong_roc_test(y_true, p1, p2):
    """Two-sided paired DeLong p for AUC(p1) vs AUC(p2); returns (auc1, auc2, p)."""
    y_true = np.asarray(y_true)
    order = (-y_true).argsort()
    m = int(y_true.sum())
    ps = np.vstack((p1, p2))[:, order]
    aucs, cov = _fast_delong(ps, m)
    l = np.array([[1, -1]])
    var = (l @ cov @ l.T).item()
    z = np.abs(aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * scipy.stats.norm.sf(z)
    return aucs[0], aucs[1], p
