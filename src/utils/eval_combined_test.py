# =============================================================================
# eval_combined_test.py
# SECONDARY evaluation of each model on three test sets:
#   REAL      — held-out PTB-XL patients
#   SYNTH     — held-out NeuroKit2 (never trained on)
#   COMBINED  — balanced 50:50 real:synthetic (synthetic matched to real per class)
# For every model x test set it writes the FULL results suite (like a normal run
# folder): metrics_summary.csv, classification_report.csv, confusion_matrix.csv,
# a high-DPI large-text confusion_matrix.png, roc_curves.png, pr_curves.png.
# Predictions are pooled across the 3 folds. No retraining, no leakage.
# Output: outputs/results/secondary_test/<MODEL>__<TESTSET>/
# =============================================================================
import sys, os, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src")); sys.path.insert(0, str(BASE / "src" / "training"))
import train as T
from utils import threshold_optimize as TO

V4 = "expv_C_ptbxl_imagen_neurokit2_img512_bs32_e25_sc500"
CLASSES = T.CLASSES
OUTROOT = BASE / "outputs" / "results" / "secondary_test"

# (label, run-stem, arch). Add B1 / ConvNeXt once trained.
MODELS = [
    ("A_realonly",   "exp_A_ptbxl_only_img512_bs32_e25",                    "b0"),
    ("B_diffusion",  "exp_B_ptbxl_imagen_img512_bs32_e25",                  "b0"),
    ("V4_capped",    "expv_C_ptbxl_imagen_neurokit2_img512_bs32_e25_sc500", "b0"),
    ("B1_A_realonly",  "expv_A_ptbxl_only_img512_bs32_e25_b1",              "b1"),
    ("B1_B_diffusion", "expv_B_ptbxl_imagen_img512_bs32_e25_b1",            "b1"),
    ("B1_V4_capped",   "expv_C_ptbxl_imagen_neurokit2_img512_bs32_e25_b1_sc500", "b1"),
]


def unseen_nk(fold):
    return set(os.path.basename(p) for p in
               pd.read_csv(BASE / "data" / "splits" / f"{V4}_fold{fold}" / "train.csv")
               .query("source=='neurokit2'")["filepath"])


def matched_synth(fold, real_counts):
    used = unseen_nk(fold); rows = []
    for c in CLASSES:
        files = sorted(glob.glob(str(BASE / "data" / "rendered" / "neurokit2" / c / "**" / "*.png"), recursive=True))
        held = [f for f in files if os.path.basename(f) not in used]
        rows += [{"filepath": f, "label": c, "source": "neurokit2"} for f in held[:real_counts[c]]]
    return pd.DataFrame(rows)


def hi_dpi_cm(cm, title, path):
    pct = cm / cm.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(10, 8.6))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{pct[i, j]:.1f}", ha="center", va="center", fontsize=30,
                    fontweight="bold", color="white" if pct[i, j] > 55 else "#0d2136")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES, fontsize=26); ax.set_yticklabels(CLASSES, fontsize=26)
    ax.set_xlabel("Predicted", fontsize=28, fontweight="bold", labelpad=12)
    ax.set_ylabel("True", fontsize=28, fontweight="bold", labelpad=12)
    ax.set_title(title, fontsize=24, fontweight="bold", pad=16)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=18)
    cb.set_label("% of true class", fontsize=20)
    plt.tight_layout(); fig.savefig(str(path) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(path) + ".pdf", bbox_inches="tight"); plt.close(fig)


def write_suite(labels, preds, probs, out_dir, title):
    out_dir.mkdir(parents=True, exist_ok=True)
    scalars, cm, report, roc_aucs, pr_aucs, roc_data = T.compute_all_metrics(
        labels, preds, probs, run_name=f"secondary_test/{out_dir.name}")
    # relocate the files train.compute_all_metrics wrote (RESULTS_DIR/secondary_test/<name>)
    T.save_roc_curves(roc_data, roc_aucs, f"secondary_test/{out_dir.name}")
    T.save_pr_curves(labels, probs, pr_aucs, f"secondary_test/{out_dir.name}")
    hi_dpi_cm(cm, title, out_dir / "confusion_matrix")
    return scalars


def main():
    summary = []
    for name, stem, arch in MODELS:
        pools = {k: {"y": [], "p": [], "pr": []} for k in ("REAL", "SYNTH", "COMBINED")}
        ok = False
        for fold in range(3):
            mp = T.MODELS_DIR / f"{stem}_fold{fold}.pth"
            sp = BASE / "data" / "splits" / f"{stem}_fold{fold}"
            if not (mp.exists() and sp.exists()):
                continue
            ok = True
            real = pd.read_csv(sp / "test.csv"); real = real[real.source == "ptbxl"].reset_index(drop=True)
            rc = {c: int((real.label == c).sum()) for c in CLASSES}
            comb = pd.concat([real, matched_synth(fold, rc)]).reset_index(drop=True)
            ck = torch.load(mp, map_location=T.DEVICE)
            m = T.build_model(arch); m.load_state_dict(ck["state_dict"]); m.to(T.DEVICE)
            probs, labels = TO.get_probs(m, comb); preds = probs.argmax(1); del m; torch.cuda.empty_cache()
            nr = len(real)
            for key, lo, hi in [("REAL", 0, nr), ("SYNTH", nr, len(comb)), ("COMBINED", 0, len(comb))]:
                pools[key]["y"].append(labels[lo:hi]); pools[key]["p"].append(preds[lo:hi]); pools[key]["pr"].append(probs[lo:hi])
        if not ok:
            print(f"[skip] {name} (models not found)"); continue
        for key in ("REAL", "SYNTH", "COMBINED"):
            y = np.concatenate(pools[key]["y"]); p = np.concatenate(pools[key]["p"]); pr = np.concatenate(pools[key]["pr"])
            sc = write_suite(y, p, pr, OUTROOT / f"{name}__{key}", f"{name} — {key} test")
            sc["model"] = name; sc["testset"] = key; summary.append(sc)
            print(f"{name:<16}{key:<10} macroF1={f1_score(y,p,average='macro',zero_division=0):.4f}")
    pd.DataFrame(summary).to_csv(OUTROOT / "secondary_summary.csv", index=False)
    print(f"\nSaved suites + {OUTROOT / 'secondary_summary.csv'}")


if __name__ == "__main__":
    main()
