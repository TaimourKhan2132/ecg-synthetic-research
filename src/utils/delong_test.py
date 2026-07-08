# =============================================================================
# delong_test.py — DeLong test for correlated ROC-AUCs (reviewer comment #1).
#
# DeLong's test applies to ROC-AUC (not F1). For each class (one-vs-rest) we
# compare the AUC of the augmented models (B, C) against the real-only baseline
# (A) on the *same* held-out real samples, so the comparison is paired.
#
# Probabilities are recovered by re-running inference from the saved canonical
# checkpoints on each fold's exact held-out test set (filepaths taken from each
# run's predictions_with_confidence.csv). The 3 patient-grouped folds are
# disjoint, so pooling them gives one paired out-of-fold prediction per real
# sample for every model — exactly what a paired DeLong needs.
#
# Exp C uses the capped-500 run (expv_C_..._sc500), matching the paper's Exp C.
#
# Fast DeLong implementation: Sun & Xu, IEEE SPL 2014.
# =============================================================================
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[2]
RES = BASE / "outputs" / "results"
MODELS = BASE / "outputs" / "models"
OUT = RES / "delong"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["NORM", "MI", "AFIB", "TACHY"]
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# canonical seed-42 B0 runs; C = capped-500 (paper's Exp C)
RUNS = {
    "A": "exp_A_ptbxl_only_img512_bs32_e25",
    "B": "exp_B_ptbxl_imagen_img512_bs32_e25",
    "C": "expv_C_ptbxl_imagen_neurokit2_img512_bs32_e25_sc500",
}

_val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _resolve(fp):
    p = Path(fp)
    if p.exists():
        return p
    # remap absolute path recorded at train time -> this repo
    s = str(fp).replace("\\", "/")
    if "data/rendered/" in s:
        return BASE / ("data/rendered/" + s.split("data/rendered/", 1)[1])
    return p


def build_model():
    m = models.efficientnet_b0(weights=None)
    inf = m.classifier[1].in_features
    m.classifier = torch.nn.Sequential(torch.nn.Dropout(0.4),
                                        torch.nn.Linear(inf, len(CLASSES)))
    return m.to(DEVICE)


@torch.no_grad()
def infer_run(stem):
    """Return DataFrame indexed by filepath with true_idx + prob columns, pooled over 3 folds."""
    rows = []
    for fold in range(3):
        ckpt = MODELS / f"{stem}_fold{fold}.pth"
        pred_csv = RES / f"{stem}_fold{fold}" / "predictions_with_confidence.csv"
        assert ckpt.exists(), f"missing checkpoint {ckpt}"
        assert pred_csv.exists(), f"missing {pred_csv}"
        state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model = build_model()
        model.load_state_dict(state["state_dict"])
        model.eval()

        df = pd.read_csv(pred_csv)
        batch, meta = [], []
        for fp, tl in zip(df["filepath"], df["true_label"]):
            img = Image.open(_resolve(fp)).convert("RGB")
            batch.append(_val_tf(img))
            meta.append((fp, CLASSES.index(tl)))
            if len(batch) == 32:
                probs = F.softmax(model(torch.stack(batch).to(DEVICE)), dim=1).cpu().numpy()
                for (fp2, ti), pr in zip(meta, probs):
                    rows.append((fp2, ti, *pr))
                batch, meta = [], []
        if batch:
            probs = F.softmax(model(torch.stack(batch).to(DEVICE)), dim=1).cpu().numpy()
            for (fp2, ti), pr in zip(meta, probs):
                rows.append((fp2, ti, *pr))
        print(f"  [{stem} fold{fold}] {len(df)} samples")
    out = pd.DataFrame(rows, columns=["filepath", "true"] + [f"p_{c}" for c in CLASSES])
    return out.set_index("filepath")


# ---- fast DeLong (Sun & Xu 2014) --------------------------------------------
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


def delong_test(y_true, p1, p2):
    """Two-sided paired DeLong p for AUC(p1) vs AUC(p2); returns (auc1, auc2, p)."""
    order = (-y_true).argsort()
    m = int(y_true.sum())
    ps = np.vstack((p1, p2))[:, order]
    aucs, cov = _fast_delong(ps, m)
    l = np.array([[1, -1]])
    var = (l @ cov @ l.T).item()
    z = np.abs(aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * scipy.stats.norm.sf(z)
    return aucs[0], aucs[1], p


def main():
    print(f"DeLong test — device={DEVICE}")
    data = {}
    for k, v in RUNS.items():
        cache = OUT / f"probs_{k}.csv"
        if cache.exists():
            data[k] = pd.read_csv(cache, index_col="filepath")
            print(f"  [{k}] loaded cached probabilities ({len(data[k])})")
        else:
            data[k] = infer_run(v)
            data[k].to_csv(cache)
    # align on common filepaths (identical CV split across A/B/C)
    common = sorted(set(data["A"].index) & set(data["B"].index) & set(data["C"].index))
    for k in data:
        data[k] = data[k].loc[common]
    y = data["A"]["true"].to_numpy()
    assert (data["B"]["true"].to_numpy() == y).all() and (data["C"]["true"].to_numpy() == y).all(), \
        "label mismatch across runs"
    print(f"pooled paired samples: {len(common)}")

    recs = []
    for ci, cls in enumerate(CLASSES):
        yk = (y == ci).astype(int)
        pA = data["A"][f"p_{cls}"].to_numpy()
        pB = data["B"][f"p_{cls}"].to_numpy()
        pC = data["C"][f"p_{cls}"].to_numpy()
        aA, aB, pBA = delong_test(yk, pA, pB)
        _, aC, pCA = delong_test(yk, pA, pC)
        recs.append(dict(cls=cls, auc_A=aA, auc_B=aB, auc_C=aC,
                         p_BvsA=pBA, p_CvsA=pCA))
    dfres = pd.DataFrame(recs)
    macro = dict(cls="MACRO", auc_A=dfres.auc_A.mean(), auc_B=dfres.auc_B.mean(),
                 auc_C=dfres.auc_C.mean(), p_BvsA=np.nan, p_CvsA=np.nan)
    dfres = pd.concat([dfres, pd.DataFrame([macro])], ignore_index=True)

    # sanity: sklearn macro one-vs-rest AUC per model
    for k in RUNS:
        yb = np.eye(len(CLASSES))[y]
        P = data[k][[f"p_{c}" for c in CLASSES]].to_numpy()
        print(f"  sklearn macro AUC {k}: {roc_auc_score(yb, P, average='macro'):.4f}")

    dfres.to_csv(OUT / "delong_auc_test.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + dfres.to_string(index=False))
    print(f"\nsaved {OUT/'delong_auc_test.csv'}")


if __name__ == "__main__":
    main()
