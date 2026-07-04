# =============================================================================
# run_thresholds.py  (H3 driver)
# Applies per-class threshold optimization (val-tuned -> test) to every run,
# uniformly, and reports argmax vs tuned macro-F1 with paired B-A / V4-A stats.
# Waits for training to release the GPU, then runs (inference only, no retrain).
# =============================================================================
import sys, time, subprocess
from pathlib import Path
import numpy as np, pandas as pd
import torch
from sklearn.metrics import f1_score

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / "src")); sys.path.insert(0, str(BASE / "src" / "training"))
import train as T
from utils import threshold_optimize as TO

# run_name templates per (experiment, seed)
STEMS = {
    ("A", 42): "exp_A_ptbxl_only_img512_bs32_e25",
    ("A", 1):  "expv_A_ptbxl_only_img512_bs32_e25_s1",
    ("A", 2):  "expv_A_ptbxl_only_img512_bs32_e25_s2",
    ("B", 42): "exp_B_ptbxl_imagen_img512_bs32_e25",
    ("B", 1):  "expv_B_ptbxl_imagen_img512_bs32_e25_s1",
    ("B", 2):  "expv_B_ptbxl_imagen_img512_bs32_e25_s2",
    ("V4", 42): "expv_C_ptbxl_imagen_neurokit2_img512_bs32_e25_sc500",
    ("V4", 1):  "expv_C_ptbxl_imagen_neurokit2_img512_bs32_e25_s1_sc500",
    ("V4", 2):  "expv_C_ptbxl_imagen_neurokit2_img512_bs32_e25_s2_sc500",
    ("C", 42): "exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25",   # full C, canonical only
}


def gpu_busy():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Where-Object { $_.CommandLine -match 'run_finish|run_seeds|train\\.py' } | "
            "Measure-Object).Count"], capture_output=True, text=True, timeout=30).stdout.strip()
        return out.isdigit() and int(out) > 0
    except Exception:
        return False


def eval_run(run):
    mp = T.MODELS_DIR / f"{run}.pth"
    vfp = BASE / "data" / "splits" / run / "val.csv"
    tfp = BASE / "data" / "splits" / run / "test.csv"
    if not (mp.exists() and vfp.exists() and tfp.exists()):
        return None
    ckpt = torch.load(mp, map_location=T.DEVICE)
    model = T.build_model(); model.load_state_dict(ckpt["state_dict"]); model.to(T.DEVICE)
    vp, vl = TO.get_probs(model, pd.read_csv(vfp))
    tp, tl = TO.get_probs(model, pd.read_csv(tfp))
    thr = TO.tune_thresholds(vp, vl)
    argmax = f1_score(tl, np.argmax(tp, 1), average="macro", zero_division=0)
    tuned  = f1_score(tl, np.argmax(tp / thr, 1), average="macro", zero_division=0)
    del model; torch.cuda.empty_cache()
    return argmax, tuned


def main():
    print("Waiting for training to release the GPU...", flush=True)
    while gpu_busy():
        time.sleep(60)
    print("GPU free — running threshold optimization.\n", flush=True)

    rows = []
    for (exp, seed), stem in STEMS.items():
        for fold in range(3):
            run = f"{stem}_fold{fold}"
            r = eval_run(run)
            if r is None:
                print(f"[skip] {run}"); continue
            rows.append({"exp": exp, "seed": seed, "fold": fold,
                         "argmax": r[0], "tuned": r[1], "delta": r[1] - r[0]})
            print(f"{exp:3s} s{seed} f{fold}:  argmax={r[0]:.4f}  tuned={r[1]:.4f}  "
                  f"delta={r[1]-r[0]:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(BASE / "outputs" / "results" / "threshold_optimization_summary.csv", index=False)

    print("\n=== per-experiment (mean over estimates) ===")
    for exp in ["A", "B", "V4", "C"]:
        s = df[df.exp == exp]
        if len(s):
            print(f"  {exp:3s}: argmax={s.argmax.mean():.4f}  tuned={s.tuned.mean():.4f}  "
                  f"delta={s.delta.mean():+.4f}  (n={len(s)})")

    # paired B-A / V4-A on TUNED, keyed by (seed,fold)
    from scipy import stats
    def keyed(exp, col):
        s = df[df.exp == exp]
        return {(r.seed, r.fold): getattr(r, col) for r in s.itertuples()}
    print("\n=== paired vs A on TUNED macro-F1 ===")
    for exp in ["B", "V4"]:
        a, b = keyed("A", "tuned"), keyed(exp, "tuned")
        k = sorted(set(a) & set(b))
        x = np.array([b[i] for i in k]); y = np.array([a[i] for i in k]); d = x - y
        if len(d) >= 2:
            _, p = stats.ttest_rel(x, y)
            print(f"  {exp}-A tuned: n={len(d)}  +{d.mean():.4f}  {sum(d>0)}/{len(d)} pos  p2={p:.3f}")
    print("\nDONE. Saved outputs/results/threshold_optimization_summary.csv")


if __name__ == "__main__":
    main()
