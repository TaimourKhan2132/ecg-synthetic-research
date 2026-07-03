# =============================================================================
# render_neurokit2.py
# Generates synthetic ECG images using NeuroKit2 signal simulation.
# Identical visual style to render_ptbxl.py — no text labels anywhere.
# Output: data/rendered/neurokit2/{CLASS}/
# Reproducible: each image is seeded deterministically from (GLOBAL_SEED,
# condition, index). Generates 1500/class (exactly what training uses).
# =============================================================================

import neurokit2 as nk
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import pandas as pd
import os
import random
import hashlib
import warnings
from datetime import datetime
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import scipy.signal as signal
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

SAMPLING_RATE = 500
DURATION      = 10

# Generate exactly the number used in training (train.py caps NK2 at 1500/class).
# Generating the used amount directly makes selection fully deterministic and
# avoids ~10k wasted renders. BATCH_SIZE must divide TOTAL evenly.
TOTAL_IMAGES_PER_CLASS = 1500
BATCH_SIZE             = 500

# Base seed for reproducible generation. Each image is seeded deterministically
# from (GLOBAL_SEED, condition, index) inside the worker, so the full dataset is
# reproducible regardless of multiprocessing scheduling.
GLOBAL_SEED = 42

DPI = 150

OUTPUT_BASE   = Path(r"C:\Users\taimo\OneDrive\Documents 1\work\ecg-synthetic-research\data\rendered\neurokit2")
METADATA_PATH = OUTPUT_BASE / "metadata.csv"

CLASSES = ["NORM", "MI", "AFIB", "TACHY"]   # BRADY dropped

HR_RANGES = {
    "NORM":  (60, 99),
    "AFIB":  (80, 160),
    "MI":    (55, 95),
    "TACHY": (101, 180),
}

LEAD_LAYOUT = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]

# =============================================================================
# VISUAL CONSTANTS — identical to render_ptbxl.py
# =============================================================================

PAPER_BG     = "#FFF8E7"
GRID_MINOR   = "#FFAAAA"
GRID_MAJOR   = "#FF5555"
SIGNAL_COLOR = "#000000"
LEAD_COLOR   = "#222222"

# =============================================================================
# SIGNAL SIMULATION
# =============================================================================

def apply_rsa(hr_mean, duration):
    n_beats     = int((duration + 5) * (hr_mean / 60.0))
    t_approx    = np.arange(n_beats) * (60.0 / hr_mean)
    resp_rate   = random.uniform(0.2, 0.4)
    hr_fluct    = random.uniform(2, 6)
    hrs         = hr_mean + hr_fluct * np.sin(2 * np.pi * resp_rate * t_approx)
    return 60.0 / hrs


def generate_warped_ecg(mean_hr, rr_intervals, duration, sr):
    regular_hr          = int(np.round(mean_hr))
    total_required_time = max(duration, np.sum(rr_intervals)) + 5.0
    clean_ecg           = nk.ecg_simulate(
        duration=total_required_time, sampling_rate=sr,
        heart_rate=regular_hr, method="ecgsyn", noise=0.0
    )
    t_target            = np.linspace(0, duration, int(duration * sr))
    peak_times_warped   = np.cumsum(rr_intervals)
    expected_rr         = 60.0 / regular_hr
    peak_times_regular  = np.arange(1, len(rr_intervals) + 1) * expected_rr
    x_warp              = np.insert(peak_times_warped, 0, 0.0)
    y_reg               = np.insert(peak_times_regular, 0, 0.0)
    t_mapping           = np.interp(t_target, x_warp, y_reg)
    t_regular_full      = np.linspace(0, total_required_time, len(clean_ecg))
    return np.interp(t_mapping, t_regular_full, clean_ecg)


def add_realistic_noise(sig, sr):
    n = len(sig)
    t = np.linspace(0, n / sr, n)

    white_noise  = np.random.randn(n)
    b, a         = signal.butter(2, [0.05 / (sr / 2), 0.5 / (sr / 2)],
                                 btype='bandpass')
    pink_wander  = signal.filtfilt(b, a, white_noise)
    if np.max(np.abs(pink_wander)) > 0:
        pink_wander = (pink_wander / np.max(np.abs(pink_wander))) \
                      * random.uniform(0.1, 0.3)

    emg_base     = np.random.normal(0, 1, n)
    b_emg, a_emg = signal.butter(2, 30 / (sr / 2), btype='highpass')
    emg          = signal.filtfilt(b_emg, a_emg, emg_base) \
                   * random.uniform(0.005, 0.015)

    base_freq    = random.choice([50.0, 60.0]) + random.uniform(-0.1, 0.1)
    power        = random.uniform(0.005, 0.015) \
                   * np.sin(2 * np.pi * base_freq * t)
    power       += random.uniform(0.001, 0.005) \
                   * np.sin(2 * np.pi * (base_freq * 2) * t)

    return sig + pink_wander + emg + power


def simulate_nsr(hr, duration, sr):
    return generate_warped_ecg(hr, apply_rsa(hr, duration), duration, sr)


def simulate_tachy(hr, duration, sr):
    return generate_warped_ecg(hr, apply_rsa(hr, duration), duration, sr)


def simulate_afib(hr, duration, sr):
    n_samples = int(duration * sr)
    t         = np.linspace(0, duration, n_samples)
    states    = ['short', 'normal', 'long']
    transitions = {
        'short':  [0.15, 0.60, 0.25],
        'normal': [0.40, 0.20, 0.40],
        'long':   [0.70, 0.20, 0.10],
    }
    mean_rr           = 60.0 / hr
    state_multipliers = {'short': 0.75, 'normal': 1.0, 'long': 1.35}
    rr_intervals      = []
    current_state     = random.choice(states)
    time_elapsed      = 0

    while time_elapsed < duration + 5.0:
        multiplier   = state_multipliers[current_state] * random.uniform(0.9, 1.1)
        current_rr   = np.clip(mean_rr * multiplier, 0.35, 1.2)
        rr_intervals.append(current_rr)
        time_elapsed += current_rr
        current_state = np.random.choice(states, p=transitions[current_state])

    base_ecg = generate_warped_ecg(hr, rr_intervals, duration, sr)

    try:
        _, rpeaks = nk.ecg_peaks(base_ecg, sampling_rate=sr)
        r_peaks   = rpeaks['ECG_R_Peaks']
        for peak in r_peaks:
            p_start = max(5, peak - int(0.22 * sr))
            p_end   = max(5, peak - int(0.04 * sr))
            if p_start > 0 and (p_end - p_start) > 5:
                x_anchor = [p_start - 5, p_start, p_end, p_end + 5]
                y_anchor = [base_ecg[i] for i in x_anchor]
                cs       = CubicSpline(x_anchor, y_anchor)
                base_ecg[p_start:p_end] = cs(np.arange(p_start, p_end))
            qrs_start = max(0, peak - int(0.05 * sr))
            qrs_end   = min(n_samples, peak + int(0.05 * sr))
            base_ecg[qrs_start:qrs_end] *= random.uniform(0.8, 1.2)
    except Exception:
        pass

    f1, f2   = random.uniform(4, 6), random.uniform(6, 8)
    f_wave   = (random.uniform(0.03, 0.06)
                * np.sin(2 * np.pi * f1 * t + random.uniform(0, 2 * np.pi))
                + random.uniform(0.01, 0.04)
                * np.sin(2 * np.pi * f2 * t + random.uniform(0, 2 * np.pi)))
    return base_ecg + f_wave


def simulate_mi(hr, duration, sr):
    base_ecg = generate_warped_ecg(hr, apply_rsa(hr, duration), duration, sr)
    st_mask  = np.zeros_like(base_ecg)
    q_mask   = np.zeros_like(base_ecg)

    try:
        _, rpeaks = nk.ecg_peaks(base_ecg, sampling_rate=sr)
        r_peaks   = rpeaks['ECG_R_Peaks']
        for peak in r_peaks:
            j_point  = peak + int(0.06 * sr)
            st_end   = peak + int(0.24 * sr)
            if st_end < len(base_ecg):
                seg_len    = st_end - j_point
                t_seg      = np.linspace(0, 1, seg_len)
                st_max     = random.uniform(0.15, 0.45)
                curve_type = random.choice(['convex', 'concave'])
                curve      = (st_max * (1 - t_seg ** 2) if curve_type == 'convex'
                              else st_max * (1 - (t_seg - 1) ** 2))
                st_mask[j_point:st_end] = curve
            q_start = max(0, peak - int(0.06 * sr))
            q_end   = max(0, peak - int(0.02 * sr))
            if q_end > q_start:
                q_len   = q_end - q_start
                q_depth = random.uniform(0.05, 0.25)
                q_curve = np.concatenate([
                    np.linspace(0, -q_depth, q_len // 2),
                    np.linspace(-q_depth, 0, q_len - q_len // 2)
                ])
                q_mask[q_start:q_end] = q_curve
    except Exception:
        pass

    return {
        "base":    base_ecg,
        "st_mask": st_mask,
        "q_mask":  q_mask,
        "region":  random.choice(["Anterior", "Inferior", "Lateral"]),
    }


def simulate_signal(condition):
    for attempt in range(5):
        try:
            hr          = random.randint(*HR_RANGES[condition])
            simulators  = {
                "NORM":  simulate_nsr,
                "TACHY": simulate_tachy,
                "AFIB":  simulate_afib,
                "MI":    simulate_mi,
            }
            signal_out  = simulators[condition](hr, DURATION, SAMPLING_RATE)
            check_sig   = (signal_out["base"]
                           if isinstance(signal_out, dict) else signal_out)
            if np.var(check_sig) < 0.005:
                raise ValueError("Degenerate signal.")
            if condition == "AFIB":
                _, rpeaks = nk.ecg_peaks(check_sig, sampling_rate=SAMPLING_RATE)
                rr        = np.diff(rpeaks['ECG_R_Peaks'])
                if len(rr) > 2 and np.std(rr) / np.mean(rr) < 0.06:
                    raise ValueError("AFIB too regular.")
            return signal_out, hr
        except Exception as e:
            if attempt == 4:
                raise RuntimeError(f"Failed {condition}") from e

# =============================================================================
# 12-LEAD DERIVATION
# =============================================================================

def derive_12_leads(base_signal_data, condition):
    if isinstance(base_signal_data, dict):
        base_signal = base_signal_data["base"]
        st_mask     = base_signal_data["st_mask"]
        q_mask      = base_signal_data["q_mask"]
        region      = base_signal_data["region"]
    else:
        base_signal = base_signal_data
        st_mask = q_mask = 0
        region  = None

    sr    = SAMPLING_RATE
    leads = {}

    def construct_lead(scale, sig=base_signal, add_st=0.0, add_q=0.0):
        clean = (scale * sig) + (add_st * st_mask) + (add_q * q_mask)
        return add_realistic_noise(clean, sr)

    coeff = {
        "I": 0.70, "II": 1.0,  "III": 0.40,
        "aVR": -0.50, "aVL": 0.30, "aVF": 0.60,
        "V1": -0.25, "V2": 0.15, "V3": 0.55,
        "V4": 0.85,  "V5": 0.90, "V6": 0.70,
    }

    if condition == "MI":
        st_w = {L: 0.0 for L in coeff}
        q_w  = {L: 0.0 for L in coeff}
        if region == "Anterior":
            for L in ["V1", "V2", "V3", "V4"]:
                st_w[L] = random.uniform(0.8, 1.2); q_w[L] = 1.0
            for L in ["II", "III", "aVF"]:
                st_w[L] = random.uniform(-0.6, -0.3)
        elif region == "Inferior":
            for L in ["II", "III", "aVF"]:
                st_w[L] = random.uniform(0.8, 1.2); q_w[L] = 1.0
            for L in ["I", "aVL", "V2"]:
                st_w[L] = random.uniform(-0.6, -0.3)
        elif region == "Lateral":
            for L in ["I", "aVL", "V5", "V6"]:
                st_w[L] = random.uniform(0.7, 1.1); q_w[L] = 1.0
            for L in ["II", "III", "aVF", "V1"]:
                st_w[L] = random.uniform(-0.5, -0.2)
        for L in coeff:
            leads[L] = construct_lead(coeff[L], add_st=st_w[L], add_q=q_w[L])
    else:
        for L in coeff:
            leads[L] = construct_lead(coeff[L] * random.uniform(0.85, 1.15))

    return leads

# =============================================================================
# ECG AXES — identical to render_ptbxl.py
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
# ECG RENDERER — NO text labels anywhere
# =============================================================================

def render_ecg(leads_dict, condition, hr, save_path, image_id):
    fig = plt.figure(figsize=(20, 12), facecolor=PAPER_BG)
    fig.patch.set_facecolor(PAPER_BG)

    gs = GridSpec(
        4, 4, figure=fig,
        hspace=0.08, wspace=0.04,
        top=0.98,          # changed from 0.90 — header space removed
        bottom=0.04,
        left=0.04, right=0.98
    )

    sr        = SAMPLING_RATE
    col_secs  = 2.5
    col_samps = int(col_secs * sr)

    for row_i, row_leads in enumerate(LEAD_LAYOUT):
        for col_i, lead_name in enumerate(row_leads):
            ax    = fig.add_subplot(gs[row_i, col_i])
            setup_ecg_axes(ax, x_max=col_secs)
            sig   = leads_dict[lead_name]
            start = col_i * col_samps
            end   = min(start + col_samps, len(sig))
            t     = np.linspace(0, col_secs, end - start)
            ax.plot(t, sig[start:end], color=SIGNAL_COLOR,
                    linewidth=0.9, antialiased=True)
            ax.text(0.015, 0.93, lead_name,
                    transform=ax.transAxes,
                    fontsize=9, fontweight='bold',
                    color=LEAD_COLOR, va='top')

    # Rhythm strip — Lead II, full 10 seconds
    ax_r   = fig.add_subplot(gs[3, :])
    setup_ecg_axes(ax_r, x_max=DURATION)
    t_full = np.linspace(0, DURATION, len(leads_dict["II"]))
    ax_r.plot(t_full, leads_dict["II"], color=SIGNAL_COLOR,
              linewidth=0.9, antialiased=True)

    # NO condition text. NO HR. NO ID. NO timestamp.

    plt.savefig(save_path, dpi=DPI, bbox_inches='tight',
                facecolor=PAPER_BG, edgecolor='none')
    plt.close(fig)

# =============================================================================
# WORKER
# =============================================================================

def worker_generate_image(args):
    i, condition, batch_num, batch_folder = args

    # Deterministic per-image seed (stable across processes/runs). Uses hashlib
    # because Python's built-in hash() is salted per-process and not reproducible.
    seed = int(
        hashlib.md5(f"{GLOBAL_SEED}_{condition}_{i}".encode()).hexdigest(), 16
    ) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)

    image_id  = f"SYN_{condition}_{i:05d}"
    filename  = f"{image_id}.png"
    save_path = os.path.join(batch_folder, filename)

    try:
        base_signal, hr = simulate_signal(condition)
        leads           = derive_12_leads(base_signal, condition)
        render_ecg(leads, condition, hr, save_path, image_id)
        return {
            "image_id":      image_id,
            "filename":      filename,
            "condition":     condition,
            "batch":         batch_num,
            "generator":     "neurokit2/scipy",
            "heart_rate":    hr,
            "duration_s":    DURATION,
            "sampling_rate": SAMPLING_RATE,
            "timestamp":     datetime.now().isoformat(),
            "status":        "generated",
            "error":         None,
        }
    except Exception as e:
        return {
            "image_id":  image_id,
            "filename":  filename,
            "condition": condition,
            "status":    "failed",
            "error":     str(e),
        }

# =============================================================================
# MAIN GENERATION
# =============================================================================

def setup_directories():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        (OUTPUT_BASE / cls).mkdir(parents=True, exist_ok=True)


def generate_dataset():
    setup_directories()

    if METADATA_PATH.exists():
        metadata_rows = pd.read_csv(METADATA_PATH).to_dict('records')
    else:
        metadata_rows = []

    total_batches = TOTAL_IMAGES_PER_CLASS // BATCH_SIZE
    max_workers   = max(1, multiprocessing.cpu_count() - 1)

    print("\n" + "=" * 60)
    print(f"NeuroKit2 ECG Generator [MULTIPROCESSING: {max_workers} Cores]")
    print("=" * 60)

    global_count = len(metadata_rows)

    try:
        for condition in CLASSES:
            condition_path  = OUTPUT_BASE / condition
            existing_files  = list(condition_path.rglob("*.png"))
            if len(existing_files) >= TOTAL_IMAGES_PER_CLASS:
                print(f"\n[SKIP] {condition} already complete "
                      f"({len(existing_files)} images).")
                continue

            print(f"\n--- Generating: {condition} ---")

            for batch_num in range(1, total_batches + 1):
                batch_folder = condition_path / f"batch_{str(batch_num).zfill(3)}"
                batch_folder.mkdir(parents=True, exist_ok=True)

                if any(
                    row.get('condition') == condition
                    and row.get('batch') == batch_num
                    for row in metadata_rows
                ):
                    print(f"  Batch {batch_num}/{total_batches} — skipped")
                    continue

                print(f"  Batch {batch_num}/{total_batches}")
                start_idx = (batch_num - 1) * BATCH_SIZE + 1
                end_idx   = batch_num * BATCH_SIZE
                tasks     = [
                    (i, condition, batch_num, str(batch_folder))
                    for i in range(start_idx, end_idx + 1)
                ]

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(worker_generate_image, task): task
                        for task in tasks
                    }
                    for future in tqdm(
                        as_completed(futures), total=len(tasks),
                        desc=condition, unit="img", ncols=75
                    ):
                        result = future.result()
                        if result["status"] == "generated":
                            result.pop("error", None)
                            metadata_rows.append(result)
                            global_count += 1
                        else:
                            print(f"\n[WARN] {result['image_id']}: "
                                  f"{result['error']}")

                pd.DataFrame(metadata_rows).to_csv(METADATA_PATH, index=False)
                print(f"  Batch {batch_num} saved.")

        print("\n" + "=" * 60)
        print(f"DONE — {global_count} images generated")
        print("=" * 60)
        return pd.DataFrame(metadata_rows)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Saving progress...")
        if metadata_rows:
            pd.DataFrame(metadata_rows).to_csv(METADATA_PATH, index=False)
            print(f"Saved {len(metadata_rows)} records.")
        return pd.DataFrame(metadata_rows)


if __name__ == "__main__":
    # Seeding now happens per-image inside worker_generate_image() for
    # reproducibility (see GLOBAL_SEED). No global RNG reseed needed here.
    df = generate_dataset()
    if not df.empty:
        print("\nDataset Summary:")
        print(df.groupby("condition")["image_id"].count())