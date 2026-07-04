# =============================================================================
# run_seeds.py — repeated-CV seeds to firm up the paired A/B/V4 significance.
# Runs A, B, and V4 (C capped at 500 synth/class) at 2 extra seeds x 3 folds.
# Seeds != 42 use StratifiedGroupKFold(shuffle) -> genuinely different partitions,
# so with seed 42 (already done) we get 3 seeds x 3 folds = 9 paired estimates.
# Waits for the variant + follow-up jobs to release the GPU, then runs.
# Ordered seed-first so the 2-seed analysis is ready after ~seed 1.
# =============================================================================
import subprocess, sys, time
from pathlib import Path

BASE  = Path(__file__).parent
TRAIN = BASE / "src" / "training" / "train.py"
PY    = sys.executable

EXPS = [
    ("A",  ["--experiment", "A"]),
    ("B",  ["--experiment", "B"]),
    ("V4", ["--experiment", "C", "--synth-cap", "500"]),   # capped-C = the winner
]
SEEDS = [1, 2]
queue = [(f"{n}_s{s}", base + ["--seed", str(s), "--fold", str(f)])
         for s in SEEDS for n, base in EXPS for f in (0, 1, 2)]


def gpu_busy():
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'run_variants|run_followups|train\\.py' } | "
             "Measure-Object).Count"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return out.isdigit() and int(out) > 0
    except Exception:
        return False


if __name__ == "__main__":
    print("Waiting for variant + follow-up jobs to release the GPU...", flush=True)
    while gpu_busy():
        time.sleep(60)
    print("GPU free — starting repeated-seed runs.\n", flush=True)

    print(f"Total seed runs queued: {len(queue)}\n")
    for i, (name, args) in enumerate(queue, 1):
        print(f"\n{'='*60}\n[{i}/{len(queue)}] {name}  {' '.join(args)}\n{'='*60}",
              flush=True)
        r = subprocess.run([PY, str(TRAIN)] + args)
        if r.returncode != 0:
            print(f"[ERROR] {name} failed (code {r.returncode}); continuing.")
    print("\nALL SEED RUNS COMPLETE")
