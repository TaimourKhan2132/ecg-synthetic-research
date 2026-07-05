# ECG Synthetic Augmentation — Comparison Across All Runs

*All figures are 3-fold cross-validation means on a real, patient-grouped,
held-out test split. Statistical comparisons use paired tests across folds
(and, where noted, repeated-seed CV giving n=9 estimates).*

---

## Table 1 — Contaminated vs Clean results (Macro-F1)

The earlier ("contaminated") runs were graded partly on **synthetic images that
had leaked into the test set**. The corrected ("clean") runs evaluate on a
**100% real, patient-grouped held-out test**, with all synthetic images confined
to training. The distortion scales with how much synthetic sat in each test set.

| Experiment | Synthetic in test | Old (contaminated) | Clean (real-only test) | Change |
|---|---|---|---|---|
| A — real only | 0% | 0.8661 | 0.8693 | ≈ 0 |
| B — + diffusion | ~12% | 0.8778 | 0.8865 | ≈ 0 |
| **C — + diffusion + simulation** | **~60%** | **0.9528** | **0.8851** | **−0.068** |

**A and B are essentially unchanged; only C collapses** — its headline 0.95 was
an artifact of a mostly-synthetic test set, not real performance.

---

## Table 2 — Clean results and the augmentation effect (paired, n = 9)

| Config | Macro-F1 | TACHY-F1 | Gain vs A (paired) |
|---|---|---|---|
| **A** — baseline (real only) | 0.8693 | 0.8001 | — |
| **B** — + diffusion | 0.8865 | 0.8411 | **+1.65% Macro-F1 (p = 0.005)** · **+3.05% TACHY (p = 0.002)** |
| **C** — + diffusion + full simulation | 0.8851 | 0.8282 | ≈ B — no gain (simulation adds nothing over diffusion) |
| **V4** — + diffusion + *controlled* simulation | **0.8921** | 0.8365 | **+1.89% Macro-F1 (p < 0.001)** · **+2.32% TACHY (p = 0.005)** |

---

## Table 3 — Diagnostic variants (clean, 3-fold mean)

| Variant | What it tests | Macro-F1 | TACHY-F1 |
|---|---|---|---|
| V4 — reduce synthetic ratio | too much synthetic swamps training | **0.8921** | 0.8365 |
| V2 — real-frequency class weights (full C) | class-weight confound | 0.8853 | 0.8342 |
| V1 — combined fixes (weights + ratio + minority) | do all three at once | 0.8725 | 0.8054 |
| V3 — real-frequency weights on B | weights on diffusion only | 0.8633 | 0.7989 |

---

## Table 4 — Calibration (Expected Calibration Error, lower = better)

| A | B | C |
|---|---|---|
| 0.1054 | 0.0981 | **0.0378** |

Combined augmentation yields ~**3× better calibration** — a clean, unambiguous benefit.

---

## Summary

1. **Corrected, C = B = A on Macro-F1.** The "combined augmentation delivers
   large gains" claim does not hold; C's apparent advantage was an evaluation artifact.
2. **Diffusion augmentation (B) significantly beats the baseline** — +1.65% Macro-F1,
   +3.05% on the scarce TACHY class — replicated across 9 estimates.
3. **Physiological simulation adds nothing over diffusion** (C ≈ B); however, a
   **controlled synthetic ratio (V4) is the best configuration** and significantly
   beats baseline (+1.89% Macro-F1, p < 0.001). Over-augmentation (full C)
   destabilises training and erases the gain.
4. **Calibration improves markedly** with combined augmentation.

*Architecture-robustness runs (EfficientNet-B1, ConvNeXt-Tiny) are in progress and
will be added when complete.*
