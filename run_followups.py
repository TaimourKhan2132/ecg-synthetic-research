# =============================================================================
# run_followups.py — "how to use synthetic" arms (H13 / H16 / H15), Exp C base.
# Waits for the variant job to finish (single 6GB GPU) then runs, so it can be
# launched immediately and chains automatically.
#
# H13 two-stage : train last 8 epochs on REAL only  (--finetune-real-epochs 8)
# H16 synth-lw  : down-weight synthetic in the loss  (--synth-loss-weight 0.3)
# H15 freeze-bn : freeze BatchNorm running stats      (--freeze-bn)
# Ordered fold-0-first across all arms so you get an early screen.
# =============================================================================
import subprocess, sys, time
from pathlib import Path

BASE  = Path(__file__).parent
TRAIN = BASE / "src" / "training" / "train.py"
PY    = sys.executable

ARMS = [
    ("H13_twostage", ["--experiment", "C", "--finetune-real-epochs", "8"]),
    ("H16_synthlw",  ["--experiment", "C", "--synth-loss-weight", "0.3"]),
    ("H15_freezebn", ["--experiment", "C", "--freeze-bn"]),
]
queue = [(n, base + ["--fold", str(f)]) for f in (0, 1, 2) for n, base in ARMS]


def gpu_busy():
    """True while another train.py / run_variants process holds the GPU."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'run_variants|train\\.py' } | "
             "Measure-Object).Count"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return out.isdigit() and int(out) > 0
    except Exception:
        return False


if __name__ == "__main__":
    print("Waiting for the variant job to release the GPU...", flush=True)
    while gpu_busy():
        time.sleep(60)
    print("GPU free — starting follow-up arms.\n", flush=True)

    print(f"Total follow-up runs queued: {len(queue)}\n")
    for i, (name, args) in enumerate(queue, 1):
        print(f"\n{'='*60}\n[{i}/{len(queue)}] {name}  {' '.join(args)}\n{'='*60}",
              flush=True)
        r = subprocess.run([PY, str(TRAIN)] + args)
        if r.returncode != 0:
            print(f"[ERROR] {name} fold failed (code {r.returncode}); continuing.")
    print("\nALL FOLLOW-UP RUNS COMPLETE")
