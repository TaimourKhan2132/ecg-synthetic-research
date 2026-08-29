# =============================================================================
# make_sample_figure.py — builds fig/synthetic_samples.png for the manuscript.
#
# Produces a 2 x 4 grid:
#   row 0 = Gemini 3 Pro Image generations (post-OCR-cleanup)
#   row 1 = NeuroKit2 simulations
#   cols  = NORM, MI, AFIB, TACHY
#
# Run from the repo root:  python make_sample_figure.py
# Then copy the output into the Overleaf project as fig/synthetic_samples.png
# =============================================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- Point these at your local rendered data (same paths train.py uses) ------
BASE_DIR = Path(r"C:\Users\taimo\OneDrive\Documents 1\work\ecg-synthetic-research")
IMAGEN_DIR   = BASE_DIR / "data" / "rendered" / "imagen_clean"
NEUROKIT_DIR = BASE_DIR / "data" / "rendered" / "neurokit2"

OUT_PATH = Path("fig/synthetic_samples.png")

CLASSES = ["NORM", "MI", "AFIB", "TACHY"]
ROWS = [
    ("Gemini 3 Pro Image", IMAGEN_DIR),
    ("NeuroKit2",          NEUROKIT_DIR),
]

# Deterministic pick: the first file in sorted order for each class. Change the
# index if a particular sample reads better at print size.
SAMPLE_INDEX = 0


def pick(src_dir: Path, cls: str) -> Path:
    cls_dir = src_dir / cls
    files = sorted(cls_dir.rglob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNGs under {cls_dir}")
    return files[min(SAMPLE_INDEX, len(files) - 1)]


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(ROWS), len(CLASSES),
        figsize=(9.0, 4.2), dpi=300,
    )

    for r, (row_label, src_dir) in enumerate(ROWS):
        for c, cls in enumerate(CLASSES):
            ax = axes[r, c]
            path = pick(src_dir, cls)
            ax.imshow(mpimg.imread(path))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#666666")
            if r == 0:
                ax.set_title(cls, fontsize=9, pad=4)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=8, labelpad=6)
            print(f"  [{row_label:>20}] {cls:<5} <- {path.name}")

    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.02,
                        wspace=0.04, hspace=0.06)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nWrote {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
