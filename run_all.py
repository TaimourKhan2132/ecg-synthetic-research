# =============================================================================
# run_all.py
# Runs all experiments and folds sequentially.
# Place in project root and run: python run_all.py
# Safe to interrupt — each completed run saves independently.
# =============================================================================

import subprocess
import sys
from pathlib import Path

BASE_DIR   = Path(__file__).parent
TRAIN      = BASE_DIR / "src" / "training" / "train.py"
CROSS      = BASE_DIR / "src" / "training" / "train_cross_domain.py"
PYTHON     = sys.executable

def run(script, args):
    cmd = [PYTHON, str(script)] + args
    label = " ".join(args)
    print(f"\n{'='*60}")
    print(f"STARTING: {label}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] {label} failed with code {result.returncode}")
        print("Continuing to next run...\n")
    else:
        print(f"\n[DONE] {label}\n")

# =============================================================================
# QUEUE — edit this list to control what runs
# =============================================================================

# Comment out any run you've already completed
queue = [

    # ── Experiment A — folds 1 and 2 (fold 0 already done) ──────────────
        # ── Experiment D — cross domain (no folds) ───────────────────────────
    (CROSS, ["--experiment", "D"]),

    # ── Experiment E — cross domain (no folds) ───────────────────────────
    (CROSS, ["--experiment", "E"]),

    (TRAIN, ["--experiment", "C", "--fold", "0"]),
    (TRAIN, ["--experiment", "A", "--fold", "1"]),
    (TRAIN, ["--experiment", "A", "--fold", "2"]),

    # ── Experiment B — folds 1 and 2 (fold 0 already done) ──────────────
    (TRAIN, ["--experiment", "B", "--fold", "1"]),
    (TRAIN, ["--experiment", "B", "--fold", "2"]),

    # ── Experiment C — all 2 folds ───────────────────────────────────────
    (TRAIN, ["--experiment", "C", "--fold", "1"]),
    (TRAIN, ["--experiment", "C", "--fold", "2"]),
]

# =============================================================================
# RUNNER
# =============================================================================

if __name__ == "__main__":
    print(f"\nTotal runs queued: {len(queue)}")
    print("Safe to interrupt — each run saves independently on completion.\n")

    for i, (script, args) in enumerate(queue, 1):
        print(f"[{i}/{len(queue)}]", end="")
        run(script, args)

    print("\n" + "="*60)
    print("ALL RUNS COMPLETE")
    print("="*60)