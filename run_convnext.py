# =============================================================================
# run_convnext.py — architecture-generalization test (ConvNeXt-Tiny).
# Does the augmentation gain (B/V4 over A) replicate on a *different, larger*
# architecture? Same protocol as B0/B1 (512px, effective batch 32, real-only
# patient-grouped test); only the backbone changes. Grad-CAM skipped.
#
# Pre-specified single design (A/B/V4 x 3 folds). NOT a config search.
# Waits for the B1 job to free the GPU, auto-probes the largest micro-batch
# that fits (effective batch kept at 32 via accumulation), then runs.
# =============================================================================
import subprocess, sys, time
from pathlib import Path

BASE = Path(__file__).parent
TRAIN = BASE / "src" / "training" / "train.py"
PY = sys.executable


def gpu_busy():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Where-Object { $_.CommandLine -match 'run_b1|train\\.py' } | "
            "Measure-Object).Count"], capture_output=True, text=True, timeout=30).stdout.strip()
        return out.isdigit() and int(out) > 0
    except Exception:
        return False


PROBE = r"""
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
chosen = 4
for bs in [8, 4]:   # both give effective batch 32 (accum 4 / 8)
    try:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[2].in_features, 4))
        m = m.cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4); sc = torch.cuda.amp.GradScaler()
        for _ in range(2):
            x = torch.randn(bs, 3, 512, 512, device='cuda'); y = torch.randint(0, 4, (bs,), device='cuda')
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                loss = F.cross_entropy(m(x), y)
            sc.scale(loss).backward(); sc.step(opt); sc.update()
        torch.cuda.synchronize()
        if torch.cuda.max_memory_reserved()/1024**3 < 5.3:
            chosen = bs; break
        del m, opt, sc
    except RuntimeError:
        torch.cuda.empty_cache(); continue
print('MICRO_BATCH=%d' % chosen)
"""


def main():
    print("Waiting for the B1 job to release the GPU...", flush=True)
    while gpu_busy():
        time.sleep(60)
    print("GPU free — probing ConvNeXt-Tiny micro-batch...", flush=True)

    out = subprocess.run([PY, "-c", PROBE], capture_output=True, text=True).stdout
    mb = 4
    for line in out.splitlines():
        if line.startswith("MICRO_BATCH="):
            mb = int(line.split("=")[1])
    print(f"Using micro-batch {mb} (effective batch 32 via accumulation)\n", flush=True)

    common = ["--arch", "convnext_tiny", "--micro-batch", str(mb), "--no-gradcam"]
    exps = [("A", ["--experiment", "A"]),
            ("B", ["--experiment", "B"]),
            ("V4", ["--experiment", "C", "--synth-cap", "500"])]
    queue = [(n, base + common + ["--fold", str(f)]) for f in (0, 1, 2) for n, base in exps]

    print(f"ConvNeXt-Tiny sweep — {len(queue)} runs\n")
    for i, (name, args) in enumerate(queue, 1):
        print(f"\n{'='*60}\n[{i}/{len(queue)}] {name} (ConvNeXt)  {' '.join(args)}\n{'='*60}", flush=True)
        r = subprocess.run([PY, str(TRAIN)] + args)
        if r.returncode != 0:
            print(f"[ERROR] {name} fold failed (code {r.returncode}); continuing.")
    print("\nCONVNEXT RUNS COMPLETE")


if __name__ == "__main__":
    main()
