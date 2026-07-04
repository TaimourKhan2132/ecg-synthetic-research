# =============================================================================
# progress.py — live view of the 3-fold re-run (reads outputs/_train_all.log)
# Usage:
#   python progress.py            # one snapshot
#   python progress.py --watch    # refresh every 15s (Ctrl-C to stop)
#   python progress.py --watch 30 # refresh every 30s
# =============================================================================
import os, re, sys, time, subprocess
from pathlib import Path

BASE = Path(__file__).parent
# Default log; override with --log <path> (e.g. outputs/_variants.log)
LOG  = BASE / "outputs" / "_train_all.log"
for i, a in enumerate(sys.argv):
    if a == "--log" and i + 1 < len(sys.argv):
        LOG = Path(sys.argv[i + 1])

EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+) \|.*VF1:([\d.]+)(\s*\[BEST\])?")
RUN_RE   = re.compile(r"Run name:\s*(exp_[A-C]_\S+_fold(\d))")
MACRO_RE = re.compile(r"Macro F1\s*:\s*([\d.]+)")
ACC_RE   = re.compile(r"Test Accuracy\s*:\s*([\d.]+)")
DONE_RE  = re.compile(r"\[DONE\] --experiment ([ABC]) --fold (\d)")


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8).stdout.strip()
        return out or "n/a"
    except Exception:
        return "n/a"


def parse():
    if not LOG.exists():
        return None, []
    text = LOG.read_text(errors="ignore").replace("\r", "\n")

    # Split log into per-run sections keyed by run_name (order preserved).
    sections, cur = [], None
    for line in text.split("\n"):
        m = RUN_RE.search(line)
        if m:
            cur = {"run": m.group(1), "fold": int(m.group(2)),
                   "epoch": 0, "tot": 25, "best_vf1": 0.0,
                   "macro_f1": None, "acc": None}
            sections.append(cur)
        if cur is None:
            continue
        em = EPOCH_RE.search(line)
        if em:
            cur["epoch"] = int(em.group(1)); cur["tot"] = int(em.group(2))
            v = float(em.group(3))
            if v > cur["best_vf1"]:
                cur["best_vf1"] = v
        mm = MACRO_RE.search(line)
        if mm:
            cur["macro_f1"] = float(mm.group(1))
        am = ACC_RE.search(line)
        if am:
            cur["acc"] = float(am.group(1))
    running = sections[-1] if sections and sections[-1]["macro_f1"] is None else None
    return running, sections


def render():
    running, sections = parse()
    by_run = {s["run"]: s for s in sections}
    lines = []
    lines.append("=" * 66)
    lines.append("  ECG 3-FOLD RE-RUN - PROGRESS")
    lines.append("=" * 66)

    done = sum(1 for s in sections if s["macro_f1"] is not None)
    lines.append(f"  Completed: {done} folds       GPU: {gpu()}")
    lines.append("-" * 66)
    lines.append(f"  {'Run':<44}{'Epoch':<8}{'MacroF1':<8}")
    lines.append("-" * 66)

    # Show whatever runs appear in this log, in order (works for canonical + variants).
    for s in sections:
        short = s["run"].replace("_img512_bs32_e25", "")
        if s["macro_f1"] is not None:
            lines.append(f"  {short:<44}{'25/25':<8}{s['macro_f1']:.4f}  DONE")
        else:
            tag = "<-- running" if s is running else ""
            lines.append(f"  {short:<44}{str(s['epoch'])+'/'+str(s['tot']):<8}"
                         f"VF1 {s['best_vf1']:.3f} {tag}")
    if not sections:
        lines.append("  (no runs started yet)")
    lines.append("=" * 66)
    if running:
        lines.append(f"  NOW: {running['run']}  epoch {running['epoch']}/{running['tot']}  "
                     f"best VF1 {running['best_vf1']:.4f}")
    elif done == 9:
        lines.append("  ALL 9 FOLDS COMPLETE.")
    lines.append("=" * 66)
    return "\n".join(lines)


def main():
    watch = "--watch" in sys.argv
    interval = 15
    for a in sys.argv[1:]:
        if a.isdigit():
            interval = int(a)
    if not watch:
        print(render()); return
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(render())
            print(f"\n  (refreshing every {interval}s — Ctrl-C to stop)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
