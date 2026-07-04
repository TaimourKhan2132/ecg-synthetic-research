# =============================================================================
# threshold_optimize.py  (H3)
# Per-class decision-threshold tuning on VALIDATION, applied to TEST.
# Guardrails for a fair, review-proof comparison:
#   * thresholds are fit on the VAL split only, never on test
#   * the SAME procedure is applied to every experiment (A/B/C/V4)
#   * both argmax and tuned Macro-F1 are reported
# Needs the trained .pth + the run's val/test splits. Re-runs inference only
# (no retraining). GPU used briefly.
#
# Usage:
#   python src/utils/threshold_optimize.py --run exp_A_ptbxl_only_img512_bs32_e25_fold0
# =============================================================================
import sys, argparse
import numpy as np, pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "src" / "training"))
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import train as T


def get_probs(model, df):
    ds = T.ECGDataset(df, T.get_transforms()[1])
    dl = DataLoader(ds, batch_size=T.MICRO_BATCH, shuffle=False, num_workers=0)
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in dl:
            with torch.autocast(device_type="cuda", enabled=T.DEVICE.type == "cuda"):
                p = torch.softmax(model(x.to(T.DEVICE)), 1)
            probs.append(p.float().cpu().numpy()); labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def tune_thresholds(val_probs, val_labels, n_grid=41):
    """Greedy per-class scaler search maximizing macro-F1: pred = argmax(prob / thr)."""
    thr = np.ones(len(T.CLASSES))
    grid = np.linspace(0.2, 2.0, n_grid)
    def macro(th):
        return f1_score(val_labels, np.argmax(val_probs / th, 1),
                        average="macro", zero_division=0)
    improved = True
    while improved:
        improved = False
        for c in range(len(T.CLASSES)):
            best_t, best_f = thr[c], macro(thr)
            for g in grid:
                cand = thr.copy(); cand[c] = g
                f = macro(cand)
                if f > best_f + 1e-6:
                    best_f, best_t = f, g
            if best_t != thr[c]:
                thr[c] = best_t; improved = True
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run_name (dir under outputs/results & data/splits)")
    args = ap.parse_args()

    model_path = T.MODELS_DIR / f"{args.run}.pth"
    val = pd.read_csv(BASE / "data" / "splits" / args.run / "val.csv")
    test = pd.read_csv(BASE / "data" / "splits" / args.run / "test.csv")

    model, *_ = None, None
    ckpt = torch.load(model_path, map_location=T.DEVICE)
    model = T.build_model(); model.load_state_dict(ckpt["state_dict"]); model.to(T.DEVICE)

    vp, vl = get_probs(model, val)
    tp, tl = get_probs(model, test)
    thr = tune_thresholds(vp, vl)

    argmax_f1 = f1_score(tl, np.argmax(tp, 1), average="macro", zero_division=0)
    tuned_f1  = f1_score(tl, np.argmax(tp / thr, 1), average="macro", zero_division=0)
    print(f"{args.run}")
    print(f"  thresholds (per-class scaler): "
          + ", ".join(f"{c}={t:.2f}" for c, t in zip(T.CLASSES, thr)))
    print(f"  Macro-F1  argmax={argmax_f1:.4f}  tuned(val->test)={tuned_f1:.4f}  "
          f"delta={tuned_f1-argmax_f1:+.4f}")
    return argmax_f1, tuned_f1


if __name__ == "__main__":
    main()
