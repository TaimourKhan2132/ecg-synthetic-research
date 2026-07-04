# =============================================================================
# run_variants.py — diagnostic experiments for why B/C gave no real F1 gain.
# All use the clean protocol (real-only val/test). Results go to expv_* dirs,
# separate from the canonical exp_A/B/C.
#
# V1 Smart : C, augment AFIB+TACHY only, synth<=500/class, real-only weights
#            (tests minority-aug + H2 ratio + H1 weights together)
# V3 Bw    : B + real-only weights                (isolates H1 on Imagen)
# V4 Ccap  : C, synth<=500/class, default weights (isolates H2 ratio)
# V2 Cw    : C + real-only weights (full synth)   (isolates H1 at full ratio)
# =============================================================================
import subprocess, sys
from pathlib import Path

TRAIN = Path(__file__).parent / "src" / "training" / "train.py"
PY = sys.executable

VARIANTS = [
    ("V1_smart", ["--experiment", "C", "--aug-classes", "AFIB,TACHY",
                  "--synth-cap", "500", "--real-weights"]),
    ("V3_Bw",    ["--experiment", "B", "--real-weights"]),
    ("V4_Ccap",  ["--experiment", "C", "--synth-cap", "500"]),
    ("V2_Cw",    ["--experiment", "C", "--real-weights"]),
]

queue = []
for name, base in VARIANTS:
    for fold in (0, 1, 2):
        queue.append((name, base + ["--fold", str(fold)]))

if __name__ == "__main__":
    print(f"Total variant runs queued: {len(queue)}\n")
    for i, (name, args) in enumerate(queue, 1):
        print(f"\n{'='*60}\n[{i}/{len(queue)}] {name}  {' '.join(args)}\n{'='*60}")
        r = subprocess.run([PY, str(TRAIN)] + args)
        if r.returncode != 0:
            print(f"[ERROR] {name} fold failed (code {r.returncode}); continuing.")
    print("\nALL VARIANT RUNS COMPLETE")
