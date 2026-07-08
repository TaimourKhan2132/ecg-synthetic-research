# Synthetic ECG Augmentation for Image-Based Arrhythmia Classification
### Detailed Findings Report (for manuscript preparation)

This report consolidates all experimental findings for the paper. All results use
a leakage-controlled, patient-grouped cross-validation protocol with a real-only
held-out test set. Figures and CSVs referenced here are collected in `to_share/`.

---

## 1. Research question

Does augmenting a real ECG image training set with **synthetic ECGs** — generated
by a text-to-image model (Gemini 3 Pro Image) and by physiological simulation
(NeuroKit2) — improve a CNN classifier for four rhythm/'morphology classes
(NORM, MI, AFIB, TACHY), relative to a real-data-only baseline?

## 2. Materials and methods

**Data.** PTB-XL (PhysioNet), rendered as clean 12-lead ECG-paper images with no
printed text or diagnostic labels. Four classes: NORM (1500), MI (1500), AFIB
(1030), TACHY (426). Synthetic sources: **diffusion/generative** images (Gemini 3
Pro Image, ~160/class, OCR-sanitised) and **physiological simulation** (NeuroKit2,
1500/class generated).

**Model.** EfficientNet-B0 (ImageNet-pretrained), 512×512 RGB input, dropout 0.4,
4-class head. Focal loss (γ=2) with inverse-frequency class weights, AdamW
(1e-4), cosine schedule, 25 epochs, effective batch 32, FP16.

**Experiments (training-set composition is the only variable):**
- **A** — real PTB-XL only (baseline)
- **B** — real + diffusion-generated
- **C** — real + diffusion + physiological simulation (NK2 at 1500/class)
- **V4** — real + diffusion + *controlled* simulation (NK2 capped at 500/class)

**Evaluation protocol.** Patient-grouped 3-fold cross-validation (no patient
appears in more than one split); **validation and test sets are 100% real
PTB-XL**; all synthetic images are confined to training. Primary metric:
macro-F1. Significance is assessed with **paired tests across folds**, and
firmed up with **repeated-seed cross-validation (n = 9 estimates)**.

---

## 3. Primary results (real held-out test)

### Table 1 — Aggregate macro-F1 and per-class F1 (3-fold mean)

| Exp | Macro-F1 | NORM | MI | AFIB | TACHY | ECE ↓ |
|---|---|---|---|---|---|---|
| **A** — real only | 0.869 | 0.910 | 0.849 | 0.918 | 0.800 | 0.105 |
| **B** — + diffusion | 0.887 | 0.909 | 0.864 | 0.932 | 0.842 | 0.098 |
| **C** — + diff + full sim | 0.885 | 0.913 | 0.872 | 0.928 | 0.828 | **0.038** |
| **V4** — + diff + capped sim | **0.892** | 0.917 | 0.881 | 0.934 | 0.837 | 0.041 |

### Table 2 — Augmentation effect vs baseline A (paired, repeated-seed CV, n = 9)

| Comparison | Δ Macro-F1 | Δ TACHY-F1 | consistency | significance |
|---|---|---|---|---|
| **B − A** | **+1.65%** | **+3.05%** | 9/9 folds positive | p = 0.005 / 0.002 |
| **V4 − A** | **+1.89%** | **+2.32%** | 9/9 folds positive | p < 0.001 / 0.005 |
| C − B | ≈ 0 | ≈ 0 | inconsistent | n.s. |

**Findings.**
1. **Diffusion augmentation (B) yields a small but statistically significant
   improvement over the real-only baseline** — +1.65% macro-F1 and +3.05% on the
   scarce TACHY class — positive in every one of nine estimates.
2. **A controlled synthetic ratio (V4) is the strongest configuration**, +1.89%
   macro-F1 (p < 0.001), and the most stable across folds.
3. **Adding physiological simulation on top of diffusion (C vs B) provides no
   additional macro-F1 benefit** (difference ≈ 0, sign-inconsistent).

## 4. Where the gains come from — class redistribution

Augmentation does **not** raise all classes uniformly; it **redistributes**
performance (row-normalised confusion, recall per class):

| True class | A recall | V4 recall | Δ |
|---|---|---|---|
| MI | 77.5% | 84.8% | **+7.3** |
| AFIB | 92.6% | 94.8% | +2.2 |
| NORM | 94.9% | 93.6% | −1.3 |
| TACHY | 90.8% | 85.7% | −5.1 |

The largest gain is on **MI — the hardest, most-confused class** (it is
frequently mistaken for other classes at the baseline). The scarce TACHY class,
already handled well by focal loss + class weighting, trades a little recall for
precision. For TACHY specifically the effect is a **precision–recall shift**
(A: P 0.72 / R 0.91 → V4: P ~0.82 / R ~0.84): synthetic augmentation makes the
model more precise but slightly less sensitive on that class.

## 5. Calibration

Combined augmentation markedly improves calibration:
**Expected Calibration Error 0.105 (A) → 0.038 (C)** — roughly a 3× reduction,
consistent across folds. The augmented models "know when they are unsure" far
better than the real-only baseline. *(Figure: calibration/ECE.)*

## 6. Synthetic-to-real ratio

The amount of synthetic data is decisive. Using the full simulation set (C, ~72%
of training synthetic) yields no gain and higher fold-to-fold variance, whereas
**capping the synthetic contribution (V4) recovers a significant, stable
improvement** (Table 2). Over-augmentation dilutes the real signal; a moderate
dose is optimal.

## 7. Secondary evaluation — real + held-out synthetic (robustness)

To probe robustness across domains, each model was additionally evaluated on a
**50:50 real + held-out-synthetic** test set (the synthetic half is NeuroKit2 the
models never trained on). This is a *secondary* result; the primary, real-world
metric remains the real-only test (Section 3).

### Table 3 — Macro-F1 by test domain (EfficientNet-B0)

| Model | REAL | SYNTHETIC (held-out) | COMBINED (50:50) |
|---|---|---|---|
| A — real only | 0.869 | 0.441 | 0.679 |
| B — + diffusion | 0.887 | 0.557 | 0.747 |
| V4 — + diff + sim | 0.892 | **0.997** | **0.945** |

**Findings.**
- Real+synthetic training keeps the real-test gain (+2.3%) **and** makes the model
  robust to synthetic-domain inputs.
- The real-only (A) and diffusion-only (B) models **fail on synthetic MI**
  (F1 = 0.06 and 0.02 respectively) — they never learned that morphology.
- The simulation-augmented model (V4) handles the held-out synthetic near-perfectly
  because it is in-distribution for the NeuroKit2 generator.
- **Interpretation caveat for the manuscript:** the high COMBINED number (0.945)
  reflects the easy in-distribution synthetic half and must be reported as a
  secondary/robustness result, never as real-world performance. *(Figure:
  `secondary_realsynth.png` — macro-F1 by domain + per-class synthetic F1.)*

## 8. Architecture generalisation (EfficientNet-B0 vs B1)

The augmentation effect is **not architecture-specific**. Repeating the study on
EfficientNet-B1 (same protocol):

| EfficientNet-B1 (3-fold) | Macro-F1 | TACHY-F1 |
|---|---|---|
| A — real only | 0.865 | 0.803 |
| B — + diffusion | 0.883 | 0.832 |
| V4 — + diff + capped sim | 0.877 | 0.818 |

**B − A = +1.8% macro-F1, positive in all 3 folds, p = 0.024** — the diffusion
gain replicates the B0 result (+1.65%). The larger B1 backbone does not improve
absolute performance (it overfits more on this data scale), consistent with the
pretrained backbones already being data-efficient.

## 9. Discussion points (for the manuscript)

- Synthetic ECG augmentation gives a **modest but genuine and replicable
  improvement**, concentrated on the hardest class (MI) and the scarcest class
  (TACHY); general-purpose diffusion images are the effective ingredient, while
  physiological simulation adds robustness to synthetic inputs but no extra
  real-test accuracy.
- The **synthetic ratio matters** — a methodological contribution: too much
  synthetic degrades stability and erases the gain.
- The benefit **generalises across backbones** (B0, B1).
- **Calibration** improves substantially with combined augmentation.

## 10. Limitations

1. Synthetic ECGs were not validated by cardiologists; some may be visually
   plausible yet physiologically imperfect.
2. TACHY rests on 426 real records — a thin base for that class.
3. NeuroKit2 TACHY corresponds to sinus tachycardia, whereas PTB-XL TACHY also
   includes supraventricular/ectopic rhythms — a distribution mismatch.
4. Backbones are ImageNet-pretrained and therefore data-efficient; the marginal
   benefit of augmentation may be understated relative to from-scratch or
   lower-data-regime training (future work).
5. Single input resolution (512×512).

---

## 11. Figures & tables provided (in `to_share/`)

- `reports/PAPER_REPORT.md` (this file), `latex_tables.tex` (copy-paste tables)
- `csv/secondary_summary.csv`, `csv/secondary_realsynth_test.csv`, `csv/confidence_intervals_3fold.csv` (each with a matching `.png`)
- Confusion matrices (high-DPI PNG + PDF): `figures/confusion_A/B/C/V4`
- `outputs/figures_paper/secondary_realsynth.(png/pdf)` — domain comparison figure
- Per-run result folders (metrics_summary.csv, classification_report.csv,
  ROC/PR curves, calibration) for A/B/C/V4 and the secondary REAL/SYNTH/COMBINED suites
- Grad-CAM figures (per-class, `gradcam/`) — correct + high-confidence, with zoomed peak region

---

## 12. Reproducibility

Environment: Python 3.11, PyTorch 2.12 (CUDA 12.x/13.x), single 6 GB GPU.
`pip install -r requirements.txt`. All runs are seeded; each fold saves
independently, so the pipeline is safe to interrupt and resume.

**End-to-end steps (training-set composition is the only variable):**

1. **Render real data** — `python src/rendering/render_ptbxl.py` (PTB-XL → label-free 512×512 ECG images).
2. **Physiological simulation** — `python src/rendering/render_neurokit2.py`.
3. **Generative images** — `python src/generation/imagen_generate.py` then `python src/rendering/sanitize_imagen.py` (OCR-clean).
4. **Patient mapping** (required before CV) — `python src/create_ptbxl_mapping.py`; leakage check via `python src/generate_leakage_report.py`.
5. **Train A / B / C** (3-fold, real-only val+test) — `python run_all.py`, or individually:
   - A: `python src/training/train.py --experiment A`
   - B: `python src/training/train.py --experiment B`
   - C (paper = capped simulation): `python src/training/train.py --experiment C --synth-cap 500`
6. **Significance (repeated-seed CV, n = 9)** — `python run_seeds.py` (2 extra seeds × 3 folds; paired tests vs A).
7. **Architecture generalisation (EfficientNet-B1)** — `python run_b1.py` (adds `--arch b1`).
8. **Domain-transfer ablations D/E** — `python src/training/train_cross_domain.py` (train on synthetic only, test on real).
9. **Secondary evaluation** (real / held-out synthetic / combined 50:50) — `python src/utils/eval_combined_test.py`.
10. **Paper figures** — `python src/utils/make_confusion_matrices.py` and `python src/utils/make_csv_graphs.py`.

Every number in this report is a 3-fold mean over the real held-out test; the
paired effects in Table 2 use the n = 9 repeated-seed estimates from step 6.
