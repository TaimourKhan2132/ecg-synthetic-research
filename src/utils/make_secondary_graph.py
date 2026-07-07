# =============================================================================
# make_secondary_graph.py — visualize secondary_realsynth_test.csv.
# (1) Macro-F1 grouped bars: each model on REAL / SYNTH(held-out) / COMBINED.
# (2) Per-class F1 on the held-out SYNTHETIC test (shows the MI collapse).
# Output: outputs/figures_paper/secondary_realsynth.png / .pdf
# =============================================================================
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
df = pd.read_csv(BASE / "outputs" / "results" / "secondary_realsynth_test.csv")
OUT = BASE / "outputs" / "figures_paper"; OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["NORM", "MI", "AFIB", "TACHY"]
TESTS = ["REAL", "SYNTH", "COMBINED"]
TCOL = {"REAL": "#1f77b4", "SYNTH": "#d62728", "COMBINED": "#2ca02c"}
models = list(dict.fromkeys(df["model"]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))

# --- Panel 1: macro-F1 grouped by model ---
x = np.arange(len(models)); w = 0.26
for i, t in enumerate(TESTS):
    vals = [df[(df.model == m) & (df.test == t)]["macro"].values[0] for m in models]
    bars = ax1.bar(x + (i - 1) * w, vals, w, label=t, color=TCOL[t], edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                 ha="center", fontsize=13, fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels([m.split("(")[0].strip() for m in models], fontsize=16)
ax1.set_ylabel("Macro-F1", fontsize=18, fontweight="bold")
ax1.set_ylim(0, 1.08)
ax1.set_title("Macro-F1: Real vs Held-out Synthetic vs Combined (50:50)",
              fontsize=17, fontweight="bold", pad=12)
ax1.legend(fontsize=15, title="Test set", title_fontsize=14, loc="lower left")
ax1.tick_params(axis="y", labelsize=13); ax1.grid(axis="y", alpha=0.3)

# --- Panel 2: per-class F1 on the SYNTHETIC (held-out) test ---
xc = np.arange(len(CLASSES)); wm = 0.26
mcol = ["#7f7f7f", "#ff7f0e", "#2ca02c"]
for i, m in enumerate(models):
    row = df[(df.model == m) & (df.test == "SYNTH")].iloc[0]
    vals = [row[c] for c in CLASSES]
    bars = ax2.bar(xc + (i - 1) * wm, vals, wm, label=m.split("(")[0].strip(),
                   color=mcol[i], edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                 ha="center", fontsize=11, fontweight="bold")
ax2.set_xticks(xc); ax2.set_xticklabels(CLASSES, fontsize=16)
ax2.set_ylabel("F1 (held-out synthetic)", fontsize=18, fontweight="bold")
ax2.set_ylim(0, 1.08)
ax2.set_title("Per-class F1 on held-out synthetic\n(real-only & diffusion-only collapse on MI)",
              fontsize=17, fontweight="bold", pad=12)
ax2.legend(fontsize=14, loc="lower left")
ax2.tick_params(axis="y", labelsize=13); ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "secondary_realsynth.png", dpi=400, bbox_inches="tight")
fig.savefig(OUT / "secondary_realsynth.pdf", bbox_inches="tight")
plt.close(fig)
print(f"saved {OUT / 'secondary_realsynth.png'} (+ .pdf)")
