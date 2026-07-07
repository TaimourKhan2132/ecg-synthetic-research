# =============================================================================
# make_confusion_matrices.py — paper-quality confusion matrices.
# Aggregates the 3-fold confusion matrices per experiment, row-normalizes to
# "% of true class", and renders LARGE-text, high-DPI figures for the paper.
# Output: outputs/figures_paper/confusion_<EXP>.png (+ .pdf)
# =============================================================================
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
RES = BASE / "outputs" / "results"
OUT = BASE / "outputs" / "figures_paper"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["NORM", "MI", "AFIB", "TACHY"]
DPI = 600

RUNS = {
    "A_baseline": "exp_A_ptbxl_only_img512_bs32_e25",
    "B_imagen":   "exp_B_ptbxl_imagen_img512_bs32_e25",
    "C_imagen_nk2": "exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25",
    "V4_capped":  "expv_C_ptbxl_imagen_neurokit2_img512_bs32_e25_sc500",
}
TITLES = {"A_baseline": "A — Real only", "B_imagen": "B — Real + Diffusion",
          "C_imagen_nk2": "C — Real + Diffusion + Simulation",
          "V4_capped": "V4 — Real + Controlled Simulation"}


def agg_cm(stem):
    tot = np.zeros((4, 4))
    for f in range(3):
        p = RES / f"{stem}_fold{f}" / "confusion_matrix.csv"
        if p.exists():
            tot += pd.read_csv(p, index_col=0).values
    return tot


def plot_cm(cm, title, path):
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(10, 8.6))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    # large in-box annotations (no % sign so big numbers fit inside the cell;
    # the colorbar already labels the unit as "% of true class")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm_pct[i, j]:.1f}", ha="center", va="center",
                    fontsize=30, fontweight="bold",
                    color="white" if cm_pct[i, j] > 55 else "#0d2136")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES, fontsize=26)
    ax.set_yticklabels(CLASSES, fontsize=26)
    ax.set_xlabel("Predicted", fontsize=28, fontweight="bold", labelpad=12)
    ax.set_ylabel("True", fontsize=28, fontweight="bold", labelpad=12)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=16)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("% of true class", fontsize=20)
    plt.tight_layout()
    fig.savefig(str(path) + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(str(path) + ".pdf", bbox_inches="tight")   # vector for LaTeX
    plt.close(fig)
    print(f"  saved {path.name}.png / .pdf")


def main():
    print(f"Rendering confusion matrices at {DPI} DPI -> {OUT}")
    for key, stem in RUNS.items():
        cm = agg_cm(stem)
        if cm.sum() == 0:
            print(f"  [skip] {key} (no data)"); continue
        plot_cm(cm, TITLES[key], OUT / f"confusion_{key}")


if __name__ == "__main__":
    main()
