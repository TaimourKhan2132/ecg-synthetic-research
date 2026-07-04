# Resume only the runs that didn't finish before the shutdown:
#   V4 (capped-C) seed 2, folds 1 and 2.
import subprocess, sys
from pathlib import Path
TRAIN = Path(__file__).parent / "src" / "training" / "train.py"
PY = sys.executable
runs = [
    ["--experiment", "C", "--synth-cap", "500", "--seed", "2", "--fold", "1"],
    ["--experiment", "C", "--synth-cap", "500", "--seed", "2", "--fold", "2"],
]
for i, args in enumerate(runs, 1):
    print(f"\n[{i}/{len(runs)}] {' '.join(args)}", flush=True)
    subprocess.run([PY, str(TRAIN)] + args)
print("\nFINISH RUNS COMPLETE")
