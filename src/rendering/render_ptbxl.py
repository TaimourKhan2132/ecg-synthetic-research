# =============================================================================
# render_ptbxl.py
# Renders PTB-XL ECG records as clean 12-lead ECG paper images.
# Style matches render_synthetic.py exactly.
# NO condition text. NO HR. NO ID. NO timestamp. NO class leakage.
# Lead names (I, II, V1...) are kept — anatomical markers, not class labels.
# Output: data/rendered/ptbxl/{CLASS}/{ecg_id}.png
# =============================================================================

import os
import ast
import wfdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_PATH = Path(r"C:\Users\taimo\Downloads\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
OUTPUT_PATH  = Path(r"C:\Users\taimo\OneDrive\Documents 1\work\ecg-synthetic-research\data\rendered\ptbxl")
METADATA_OUT = Path(r"C:\Users\taimo\OneDrive\Documents 1\work\ecg-synthetic-research\metadata\ptbxl_rendered.csv")

CLASS_CAPS = {
    "NORM":  1500,
    "MI":    1500,
    "AFIB":  9999,
    "TACHY": 9999,
}

AFIB_CODES  = {"AFIB", "AFLT"}
TACHY_CODES = {"STACH", "SVTAC", "PSVT", "BIGU", "TRIGU"}

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4",  "V5",  "V6"]

SAMPLING_RATE = 500
DURATION      = 10

# Visual constants — match render_synthetic.py exactly
PAPER_BG     = "#FFF8E7"
GRID_MINOR   = "#FFAAAA"
GRID_MAJOR   = "#FF5555"
SIGNAL_COLOR = "#000000"
LEAD_COLOR   = "#222222"
DPI          = 150
IMG_WIDTH_IN  = 20
IMG_HEIGHT_IN = 12

LEAD_LAYOUT = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]

# =============================================================================
# STEP 1 — Load PTB-XL metadata
# =============================================================================

def load_ptbxl_metadata():
    df = pd.read_csv(DATASET_PATH / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp      = pd.read_csv(DATASET_PATH / "scp_statements.csv", index_col=0)
    diag_map = scp[scp["diagnostic"] == 1.0]["diagnostic_class"].to_dict()

    return df, diag_map

# =============================================================================
# STEP 2 — Assign primary class label
# =============================================================================

def assign_label(scp_codes: dict, diag_map: dict):
    """
    Rhythm codes (AFIB/TACHY) use ANY presence — PTB-XL stores them
    with likelihood=0 as a binary flag, not a probability.
    Diagnostic codes (NORM/MI) use >= 50 confidence threshold.
    Records with multiple matching labels are excluded (conflict).
    """
    has_afib  = any(c in AFIB_CODES  for c in scp_codes)
    has_tachy = any(c in TACHY_CODES for c in scp_codes)

    confident_diag = {c for c, v in scp_codes.items() if v >= 50}
    diag_superclasses = {diag_map[c] for c in confident_diag if c in diag_map}

    has_norm = "NORM" in diag_superclasses
    has_mi   = "MI"   in diag_superclasses

    labels = []
    if has_afib:  labels.append("AFIB")
    if has_tachy: labels.append("TACHY")
    if has_norm:  labels.append("NORM")
    if has_mi:    labels.append("MI")

    return labels[0] if len(labels) == 1 else None

# =============================================================================
# STEP 3 — Build filtered, balanced dataset
# =============================================================================

def build_dataset(df, diag_map):
    records = []
    for ecg_id, row in df.iterrows():
        label = assign_label(row["scp_codes"], diag_map)
        if label is None:
            continue
        records.append({
            "ecg_id":      ecg_id,
            "label":       label,
            "filename_hr": row["filename_hr"],
        })

    dataset = pd.DataFrame(records)

    balanced = []
    print("\nClass counts before capping:")
    for cls in CLASS_CAPS:
        subset = dataset[dataset["label"] == cls]
        print(f"  {cls}: {len(subset)} available")
        cap = CLASS_CAPS[cls]
        if len(subset) > cap:
            subset = subset.sample(n=cap, random_state=42)
        balanced.append(subset)

    result = pd.concat(balanced).reset_index(drop=True)

    print("\nClass counts after capping:")
    for cls in CLASS_CAPS:
        n = len(result[result["label"] == cls])
        print(f"  {cls}: {n}")
    print(f"  TOTAL: {len(result)}")

    return result

# =============================================================================
# STEP 4 — ECG axes setup
# =============================================================================

def setup_ecg_axes(ax, x_max, y_min=-1.5, y_max=1.5):
    ax.set_facecolor(PAPER_BG)
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.04))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, which='minor', color=GRID_MINOR, linewidth=0.3, alpha=0.6)
    ax.grid(True, which='major', color=GRID_MAJOR, linewidth=0.7, alpha=0.8)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_MAJOR)
        spine.set_linewidth(0.5)

# =============================================================================
# STEP 5 — ECG renderer
# =============================================================================

def render_ecg(signal: np.ndarray, output_path: Path):
    """
    Renders a PTB-XL 12-lead ECG.
    signal: np.ndarray shape (5000, 12), in mV.
    NO condition text. NO HR. NO ID. NO timestamp.
    Lead names only — these are anatomical markers, not class labels.
    """
    fig = plt.figure(figsize=(IMG_WIDTH_IN, IMG_HEIGHT_IN), facecolor=PAPER_BG)
    fig.patch.set_facecolor(PAPER_BG)

    gs = GridSpec(
        4, 4, figure=fig,
        hspace=0.08, wspace=0.04,
        top=0.98, bottom=0.04,
        left=0.04, right=0.98
    )

    col_secs  = 2.5
    col_samps = int(col_secs * SAMPLING_RATE)

    for row_i, row_leads in enumerate(LEAD_LAYOUT):
        for col_i, lead_name in enumerate(row_leads):
            ax = fig.add_subplot(gs[row_i, col_i])
            setup_ecg_axes(ax, x_max=col_secs)

            lead_idx = LEAD_NAMES.index(lead_name)
            start    = col_i * col_samps
            end      = min(start + col_samps, signal.shape[0])
            seg      = np.clip(signal[start:end, lead_idx], -2.5, 2.5)
            t        = np.linspace(0, col_secs, end - start)

            ax.plot(t, seg, color=SIGNAL_COLOR, linewidth=0.9, antialiased=True)

            ax.text(
                0.015, 0.93, lead_name,
                transform=ax.transAxes,
                fontsize=9, fontweight='bold',
                color=LEAD_COLOR, va='top'
            )

    # Rhythm strip — Lead II, full 10 seconds
    ax_r   = fig.add_subplot(gs[3, :])
    setup_ecg_axes(ax_r, x_max=DURATION)
    t_full = np.linspace(0, DURATION, signal.shape[0])
    rhythm = np.clip(signal[:, LEAD_NAMES.index("II")], -2.5, 2.5)
    ax_r.plot(t_full, rhythm, color=SIGNAL_COLOR, linewidth=0.9, antialiased=True)

    plt.savefig(
        output_path, dpi=DPI,
        bbox_inches='tight',
        facecolor=PAPER_BG,
        edgecolor='none',
        format='png'
    )
    plt.close(fig)

# =============================================================================
# STEP 6 — Main pipeline
# =============================================================================

def main():
    print("=" * 60)
    print("PTB-XL ECG Renderer")
    print("=" * 60)

    for cls in CLASS_CAPS:
        (OUTPUT_PATH / cls).mkdir(parents=True, exist_ok=True)
    METADATA_OUT.parent.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading PTB-XL metadata...")
    df, diag_map = load_ptbxl_metadata()
    print(f"  Total records in database: {len(df)}")

    print("\n[2/4] Filtering and balancing classes...")
    dataset = build_dataset(df, diag_map)

    print("\n[3/4] Rendering ECG images...")
    metadata_rows = []
    skipped = 0
    errors  = 0

    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        ecg_id   = row["ecg_id"]
        label    = row["label"]
        filepath = DATASET_PATH / row["filename_hr"]
        out_file = OUTPUT_PATH / label / f"{ecg_id:05d}.png"

        if out_file.exists():
            skipped += 1
            continue

        try:
            record = wfdb.rdrecord(str(filepath))
            signal = record.p_signal

            if signal is None or signal.shape != (5000, 12):
                errors += 1
                continue

            render_ecg(signal, out_file)

            metadata_rows.append({
                "ecg_id":      ecg_id,
                "label":       label,
                "source":      "ptbxl",
                "file_path":   str(out_file),
                "filename_hr": row["filename_hr"],
            })

        except Exception as e:
            errors += 1
            tqdm.write(f"[SKIP] ecg_id={ecg_id}: {e}")

    print("\n[4/4] Saving metadata CSV...")
    meta_df = pd.DataFrame(metadata_rows)

    # Append to existing metadata if renderer was restarted
    if METADATA_OUT.exists():
        existing = pd.read_csv(METADATA_OUT)
        meta_df  = pd.concat([existing, meta_df]).drop_duplicates(subset="ecg_id")

    meta_df.to_csv(METADATA_OUT, index=False)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Rendered : {len(metadata_rows)}")
    print(f"  Skipped  : {skipped} (already existed)")
    print(f"  Errors   : {errors}")
    print(f"  Metadata : {METADATA_OUT}")
    print("\nFinal class distribution:")
    print(meta_df["label"].value_counts().to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()