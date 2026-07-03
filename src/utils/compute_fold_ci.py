# =============================================================================
# compute_fold_ci.py
# Aggregates the 9 metrics_summary.csv files (Exp A/B/C x fold0/1/2) into
# mean +/- 95% CI (t-distribution, df=2) for Macro F1 and per-class F1.
# Correct approach for 3-fold patient-grouped CV (not bootstrap on one fold).
#
# Usage: python src/utils/compute_fold_ci.py
# =============================================================================

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

BASE_DIR    = Path(r"C:\Users\taimo\OneDrive\Documents 1\work\ecg-synthetic-research")
RESULTS_DIR = BASE_DIR / "outputs" / "results"
CLASSES     = ["NORM", "MI", "AFIB", "TACHY"]
CONFIDENCE  = 0.95

# Matches: exp_A_ptbxl_only_img512_bs32_e25_fold0, exp_B_ptbxl_imagen_..._fold1, etc.
RUN_PATTERN = re.compile(r"^exp_([ABC])_.*_fold(\d)$", re.IGNORECASE)

EXP_LABELS = {
    "A": "Exp A — Baseline",
    "B": "Exp B — +Imagen",
    "C": "Exp C — +Imagen+NK2",
}


def discover_runs():
    runs = {"A": {}, "B": {}, "C": {}}
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results dir not found: {RESULTS_DIR}")
    for d in RESULTS_DIR.iterdir():
        if not d.is_dir():
            continue
        m = RUN_PATTERN.match(d.name)
        if m:
            exp, fold = m.group(1).upper(), int(m.group(2))
            runs[exp][fold] = d
    return runs


def load_metrics(run_dir: Path):
    """Read macro_f1 and per-class f1 directly from metrics_summary.csv."""
    ms_path = run_dir / "metrics_summary.csv"
    if not ms_path.exists():
        return {}
    df = pd.read_csv(ms_path)
    row = df.iloc[0]
    out = {"macro_f1": float(row["macro_f1"])}
    for cls in CLASSES:
        col = f"f1_{cls}"
        if col in df.columns:
            out[cls] = float(row[col])
    # bonus: keep ROC-AUC/PR-AUC too, useful for the paper narrative
    if "macro_roc_auc" in df.columns:
        out["roc_auc"] = float(row["macro_roc_auc"])
    if "macro_pr_auc" in df.columns:
        out["pr_auc"] = float(row["macro_pr_auc"])
    return out


def mean_ci_t(values, confidence=0.95):
    values = np.array(values, dtype=float)
    n = len(values)
    mean = np.mean(values)
    if n < 2:
        return mean, np.nan, np.nan
    sem = stats.sem(values)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * sem
    return mean, mean - margin, mean + margin


def main():
    runs = discover_runs()

    print("=" * 70)
    print("DISCOVERED RUNS")
    print("=" * 70)
    for exp in ["A", "B", "C"]:
        folds_found = sorted(runs[exp].keys())
        print(f"Exp {exp}: folds found = {folds_found}")
        for f, path in sorted(runs[exp].items()):
            print(f"    fold{f} -> {path.name}")
        if len(folds_found) != 3:
            print(f"    [WARNING] Expected 3 folds, found {len(folds_found)}.")
    print()

    results = []

    for exp in ["A", "B", "C"]:
        fold_metrics = {}
        for fold, path in sorted(runs[exp].items()):
            m = load_metrics(path)
            if not m:
                print(f"[SKIP] No metrics_summary.csv in {path}")
                continue
            fold_metrics[fold] = m

        if len(fold_metrics) < 2:
            print(f"[WARNING] Exp {exp} has <2 valid folds — cannot compute CI.")
            continue

        row = {"Experiment": EXP_LABELS[exp], "N_folds": len(fold_metrics)}

        macro_vals = [v["macro_f1"] for v in fold_metrics.values()]
        mean, lo, hi = mean_ci_t(macro_vals, CONFIDENCE)
        row["Macro F1 (per fold)"] = [round(v, 4) for v in macro_vals]
        row["Macro F1 Mean"] = round(mean, 4)
        row["Macro F1 CI"] = f"[{lo:.4f}, {hi:.4f}]"
        row["Macro F1 LaTeX"] = f"{mean:.3f} $\\pm$ {(hi-lo)/2:.3f}"
        row["Macro F1 LaTeX Range"] = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"

        for cls in CLASSES:
            cls_vals = [v[cls] for v in fold_metrics.values() if cls in v]
            if cls_vals:
                mean_c, lo_c, hi_c = mean_ci_t(cls_vals, CONFIDENCE)
                row[f"{cls} F1 (per fold)"] = [round(v, 4) for v in cls_vals]
                row[f"{cls} F1 Mean"] = round(mean_c, 4)
                row[f"{cls} F1 CI"] = f"[{lo_c:.4f}, {hi_c:.4f}]"

        results.append(row)

        print("-" * 70)
        print(EXP_LABELS[exp])
        print(f"  Macro F1 per fold : {row['Macro F1 (per fold)']}")
        print(f"  Macro F1 mean     : {row['Macro F1 Mean']}")
        print(f"  Macro F1 95% CI   : {row['Macro F1 CI']}")
        for cls in CLASSES:
            if f"{cls} F1 Mean" in row:
                print(f"  {cls:6s} F1 mean/CI : {row[f'{cls} F1 Mean']} {row[f'{cls} F1 CI']}")

    if not results:
        print("\nNo results computed. Check folder naming.")
        return

    out_df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "confidence_intervals_3fold.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 70)
    print("LaTeX-ready strings for Table 4 (Macro F1 mean +/- CI half-width)")
    print("=" * 70)
    for row in results:
        print(f"{row['Experiment']}: {row['Macro F1 LaTeX']}   "
              f"(range form: {row['Macro F1 LaTeX Range']})")


if __name__ == "__main__":
    main()