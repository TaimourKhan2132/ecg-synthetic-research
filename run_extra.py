# =============================================================================
# run_extra.py — finish B1 (fold 2 for A/B/V4) + 1 fold of ConvNeXt-Tiny (A/B/V4).
# Waits for the GPU to be free (eval/other jobs), then runs. Grad-CAM skipped.
# ConvNeXt uses micro-batch 8 (effective batch 32) to fit the 6GB card.
# =============================================================================
import subprocess, sys, time
from pathlib import Path

TRAIN = Path(__file__).parent / "src" / "training" / "train.py"
PY = sys.executable

EXPS = [("A", ["--experiment", "A"]),
        ("B", ["--experiment", "B"]),
        ("V4", ["--experiment", "C", "--synth-cap", "500"])]

queue = []
# B1: finish fold 2
for n, base in EXPS:
    queue.append((f"B1 {n} f2", base + ["--arch", "b1", "--no-gradcam", "--fold", "2"]))
# ConvNeXt-Tiny: fold 0
for n, base in EXPS:
    queue.append((f"ConvNeXt {n} f0",
                  base + ["--arch", "convnext_tiny", "--micro-batch", "8", "--no-gradcam", "--fold", "0"]))


def gpu_free():
    try:
        u = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        return int(u) < 800
    except Exception:
        return True


if __name__ == "__main__":
    print("Waiting for the GPU to be free...", flush=True)
    while not gpu_free():
        time.sleep(30)
    print("GPU free — running extra folds.\n", flush=True)
    for i, (name, args) in enumerate(queue, 1):
        print(f"\n{'='*60}\n[{i}/{len(queue)}] {name}  {' '.join(args)}\n{'='*60}", flush=True)
        r = subprocess.run([PY, str(TRAIN)] + args)
        if r.returncode != 0:
            print(f"[ERROR] {name} failed (code {r.returncode}); continuing.")
    print("\nEXTRA RUNS COMPLETE")
