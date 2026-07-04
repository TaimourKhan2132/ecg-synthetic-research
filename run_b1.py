# =============================================================================
# run_b1.py — robustness fork: does the augmentation gain replicate on a second
# backbone (EfficientNet-B1)? Compares A / B / V4(capped-C) on B1, 3 folds.
# Identical protocol to the B0 runs (512px, eff-batch 32, real-only test);
# only the architecture changes. Grad-CAM skipped (not needed for this check).
# =============================================================================
import subprocess, sys
from pathlib import Path

TRAIN = Path(__file__).parent / "src" / "training" / "train.py"
PY = sys.executable
COMMON = ["--arch", "b1", "--no-gradcam"]
EXPS = [
    ("A",  ["--experiment", "A"]),
    ("B",  ["--experiment", "B"]),
    ("V4", ["--experiment", "C", "--synth-cap", "500"]),
]
# fold-first so fold-0 of A/B/V4 is available early for a quick read
queue = [(n, base + COMMON + ["--fold", str(f)]) for f in (0, 1, 2) for n, base in EXPS]

if __name__ == "__main__":
    print(f"B1 robustness fork — {len(queue)} runs\n", flush=True)
    for i, (name, args) in enumerate(queue, 1):
        print(f"\n{'='*60}\n[{i}/{len(queue)}] {name} (B1)  {' '.join(args)}\n{'='*60}", flush=True)
        r = subprocess.run([PY, str(TRAIN)] + args)
        if r.returncode != 0:
            print(f"[ERROR] {name} fold failed (code {r.returncode}); continuing.")
    print("\nB1 RUNS COMPLETE")
