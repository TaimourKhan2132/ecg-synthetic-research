# =============================================================================
# make_csv_graphs.py — one high-DPI PNG per CSV in to_share/csv/, so every table
# has a matching figure sitting beside it.
# =============================================================================
import sys, shutil
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
RES = BASE / "outputs" / "results"
CSVDIR = BASE / "to_share" / "csv"
CSVDIR.mkdir(parents=True, exist_ok=True)

# ---- 1. primary macro-F1 with 95% CI (confidence_intervals_3fold.csv) ----
STEMS = {"A": "exp_A_ptbxl_only_img512_bs32_e25",
         "B": "exp_B_ptbxl_imagen_img512_bs32_e25",
         "C": "exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25",
         "V4": "expv_C_ptbxl_imagen_neurokit2_img512_bs32_e25_sc500"}
LAB = {"A": "A\nreal only", "B": "B\n+diffusion", "C": "C\n+diff+sim", "V4": "V4\n+diff+capped"}
means, errs, labels = [], [], []
for k, s in STEMS.items():
    v = np.array([pd.read_csv(RES / f"{s}_fold{f}" / "metrics_summary.csv").iloc[0]["macro_f1"] for f in range(3)])
    m = v.mean(); h = stats.t.ppf(0.975, 2) * stats.sem(v)
    means.append(m); errs.append(h); labels.append(LAB[k])
fig, ax = plt.subplots(figsize=(9, 6.5))
cols = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]
bars = ax.bar(range(4), means, yerr=errs, capsize=8, color=cols, edgecolor="black", linewidth=0.8)
for b, m in zip(bars, means):
    ax.text(b.get_x() + b.get_width() / 2, m + 0.004, f"{m:.3f}", ha="center", fontsize=15, fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=15)
ax.set_ylabel("Macro-F1 (mean ± 95% CI)", fontsize=17, fontweight="bold")
ax.set_ylim(0.83, 0.92); ax.set_title("Primary result — real held-out test (3-fold)", fontsize=18, fontweight="bold")
ax.grid(axis="y", alpha=0.3); plt.tight_layout()
fig.savefig(CSVDIR / "confidence_intervals_3fold.png", dpi=400, bbox_inches="tight"); plt.close(fig)
print("saved confidence_intervals_3fold.png")

# ---- 2. secondary_summary.csv -> B0 & B1 grouped bars by test domain ----
d = pd.read_csv(RES / "secondary_test" / "secondary_summary.csv")
TESTS = ["REAL", "SYNTH", "COMBINED"]; TCOL = {"REAL": "#1f77b4", "SYNTH": "#d62728", "COMBINED": "#2ca02c"}
groups = {"EfficientNet-B0": ["A_realonly", "B_diffusion", "V4_capped"],
          "EfficientNet-B1": ["B1_A_realonly", "B1_B_diffusion", "B1_V4_capped"]}
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, (title, ms) in zip(axes, groups.items()):
    x = np.arange(len(ms)); w = 0.26
    for i, t in enumerate(TESTS):
        vals = [d[(d.model == m) & (d.testset == t)]["macro_f1"].values[0] for m in ms]
        bb = ax.bar(x + (i - 1) * w, vals, w, label=t, color=TCOL[t], edgecolor="black", linewidth=0.5)
        for b, v in zip(bb, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["A", "B", "V4"], fontsize=16)
    ax.set_ylim(0, 1.08); ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_ylabel("Macro-F1", fontsize=16, fontweight="bold"); ax.legend(fontsize=13, title="Test domain")
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Secondary evaluation: Real / Held-out Synthetic / Combined", fontsize=19, fontweight="bold")
plt.tight_layout(); fig.savefig(CSVDIR / "secondary_summary.png", dpi=400, bbox_inches="tight"); plt.close(fig)
print("saved secondary_summary.png")

# ---- 3. reuse the existing domain graph for secondary_realsynth_test.csv ----
src = BASE / "outputs" / "figures_paper" / "secondary_realsynth.png"
if src.exists():
    shutil.copy(src, CSVDIR / "secondary_realsynth_test.png")
    print("copied secondary_realsynth_test.png")
