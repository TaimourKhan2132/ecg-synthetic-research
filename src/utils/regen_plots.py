# =============================================================================
# regen_plots.py
# Regenerates confusion matrix plots from saved CSV files.
# Run after updating save_confusion_matrix_plot in train.py.
# Usage: python src/utils/regen_plots.py
# =============================================================================

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR    = Path(r"C:\Users\taimo\OneDrive\Documents 1\work\ecg-synthetic-research")
RESULTS_DIR = BASE_DIR / "outputs" / "results"
CLASSES     = ["NORM", "MI", "AFIB", "TACHY"]

RUNS = [
    "exp_A_ptbxl_only_img512_bs32_e25_fold0",
    "exp_B_ptbxl_imagen_img512_bs32_e25_fold0",
    "exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25_fold0",
]

def regen_confusion_matrix(run_name):
    results_dir = RESULTS_DIR / run_name
    cm_csv      = results_dir / "confusion_matrix.csv"

    if not cm_csv.exists():
        print(f"  [SKIP] No confusion_matrix.csv for {run_name}")
        return

    cm    = pd.read_csv(cm_csv, index_col=0).values
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax,
        linewidths=0.5,
        vmin=0, vmax=100,
        cbar_kws={"label": "% of True Class"}
    )
    for text in ax.texts:
        text.set_text(text.get_text() + "%")

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(
        f"Confusion Matrix — {run_name.split('_')[1]}",
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        results_dir / "confusion_matrix.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"  [OK] {run_name}")


if __name__ == "__main__":
    print("Regenerating confusion matrices (% only, 300 DPI)...")
    for run in RUNS:
        regen_confusion_matrix(run)
    print("Done.")