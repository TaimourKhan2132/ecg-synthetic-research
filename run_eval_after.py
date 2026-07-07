# Waits for the GPU to free (B1 finishing), then runs the full secondary-test
# suite (REAL/SYNTH/COMBINED graphs+CSVs) for the B0 and B1 models.
import subprocess, sys, time
from pathlib import Path
PY = sys.executable; BASE = Path(__file__).parent
def gpu_free():
    try:
        u = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                           capture_output=True,text=True,timeout=20).stdout.strip().splitlines()[0]
        return int(u) < 800
    except Exception:
        return True
print("waiting for GPU (B1 to finish)...", flush=True)
while not gpu_free(): time.sleep(30)
print("GPU free - running secondary-test suite", flush=True)
subprocess.run([PY, str(BASE/"src"/"utils"/"eval_combined_test.py")])
print("SECONDARY SUITE COMPLETE", flush=True)
