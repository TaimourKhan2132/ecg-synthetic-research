# =============================================================================
# make_csv_graphs.py — one high-DPI PNG per CSV in to_share/csv/.
# =============================================================================
import shutil
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

# stems for canonical (seed 42) and seed 1/2 repeats
S42 = {"A": "exp_A_ptbxl_only_img512_bs32_e25",
       "B": "exp_B_ptbxl_imagen_img512_bs32_e25",
       "C": "exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25",
       "V4": "expv_C_ptbxl_imagen_neurokit2_img512_bs32_e25_sc500"}
SEED = {42: {"A": "exp_A_ptbxl_only_img512_bs32_e25", "B": "exp_B_ptbxl_imagen_img512_bs32_e25",
             "V4": "expv_C_ptbxl_imagen_neurokit2_img512_bs32_e25_sc500"},
        1: {"A": "expv_A_ptbxl_only_img512_bs32_e25_s1", "B": "expv_B_ptbxl_imagen_img512_bs32_e25_s1",
            "V4": "expv_C_ptbxl_imagen_neurokit2_img512_bs32_e25_s1_sc500"},
        2: {"A": "expv_A_ptbxl_only_img512_bs32_e25_s2", "B": "expv_B_ptbxl_imagen_img512_bs32_e25_s2",
            "V4": "expv_C_ptbxl_imagen_neurokit2_img512_bs32_e25_s2_sc500"}}
LAB = {"A": "A\nreal only", "B": "B\n+diffusion", "C": "C\n+diff+sim", "V4": "V4\n+diff+capped"}
COL = {"A": "#7f7f7f", "B": "#1f77b4", "C": "#ff7f0e", "V4": "#2ca02c"}


def folds3(stem, col="macro_f1"):
    return np.array([pd.read_csv(RES / f"{stem}_fold{f}" / "metrics_summary.csv").iloc[0][col] for f in range(3)])


def keyed(exp, col="macro_f1"):
    out = {}
    for s in (42, 1, 2):
        for f in range(3):
            p = RES / f"{SEED[s][exp]}_fold{f}" / "metrics_summary.csv"
            if p.exists():
                out[(s, f)] = pd.read_csv(p).iloc[0][col]
    return out


def diff_ci(x, y):
    k = sorted(set(x) & set(y)); d = np.array([x[i] - y[i] for i in k])
    return d.mean(), stats.t.ppf(0.975, len(d) - 1) * stats.sem(d)


# ---- 1. primary: absolute (honest) + paired improvement (the significance) ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
for i, k in enumerate(["A", "B", "C", "V4"]):
    v = folds3(S42[k]); m = v.mean()
    ax1.bar(i, m, color=COL[k], edgecolor="black", linewidth=0.8, width=0.66)
    ax1.scatter([i] * 3, v, color="black", zorder=3, s=34)
    ax1.text(i, v.max() + 0.003, f"{m:.3f}", ha="center", fontsize=14, fontweight="bold")
ax1.set_xticks(range(4)); ax1.set_xticklabels([LAB[k] for k in ["A", "B", "C", "V4"]], fontsize=13)
ax1.set_ylabel("Macro-F1 (real test)", fontsize=16, fontweight="bold")
ax1.set_ylim(0.83, 0.918); ax1.grid(axis="y", alpha=0.3)
ax1.set_title("Absolute performance (3-fold; dots = folds)", fontsize=15, fontweight="bold")

pairs = [("B$-$A\nmacro", diff_ci(keyed("B"), keyed("A")), COL["B"]),
         ("V4$-$A\nmacro", diff_ci(keyed("V4"), keyed("A")), COL["V4"]),
         ("B$-$A\nTACHY", diff_ci(keyed("B", "f1_TACHY"), keyed("A", "f1_TACHY")), COL["B"]),
         ("V4$-$A\nTACHY", diff_ci(keyed("V4", "f1_TACHY"), keyed("A", "f1_TACHY")), COL["V4"])]
for i, (lbl, (m, h), c) in enumerate(pairs):
    ax2.bar(i, m * 100, yerr=h * 100, capsize=8, color=c, edgecolor="black", linewidth=0.8, width=0.6)
    ax2.text(i, (m + h) * 100 + 0.12, f"+{m*100:.1f}%", ha="center", fontsize=13, fontweight="bold")
ax2.axhline(0, color="black", linewidth=1.2)
ax2.set_xticks(range(4)); ax2.set_xticklabels([p[0] for p in pairs], fontsize=12)
ax2.set_ylabel("Improvement vs A  (%, 95% CI)", fontsize=15, fontweight="bold")
ax2.set_ylim(0, None); ax2.grid(axis="y", alpha=0.3)
ax2.set_title("Augmentation effect (paired, n=9)", fontsize=15, fontweight="bold")
fig.suptitle("Primary result — real held-out test", fontsize=18, fontweight="bold")
plt.tight_layout()
fig.savefig(CSVDIR / "confidence_intervals_3fold.png", dpi=400, bbox_inches="tight"); plt.close(fig)
print("saved confidence_intervals_3fold.png (absolute + paired effect)")

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
