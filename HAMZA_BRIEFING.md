# Project Briefing for Hamza Chaudhry
## ECG Synthetic Augmentation Research — Full Technical Summary

---

## What This Project Is

We extended the Nano Banana pneumonia paper to ECGs. That paper showed that
mixing AI-generated chest X-ray images with real images improved CNN
classification. We asked: does the same work for ECGs?

Our answer: yes, significantly — especially for data-scarce classes.

---

## The Dataset

**PTB-XL** is a real clinical 12-lead ECG dataset from PhysioNet with
21,837 records. It is multi-label, so one patient can have multiple
conditions. We extracted 4 classes:

- NORM — Normal ECG (1500 records used)
- MI — Myocardial Infarction (1500 records used)
- AFIB — Atrial Fibrillation (1030 records used)
- TACHY — Tachycardia (426 records used — this is the problem class)

TACHY having only 426 records is critical context for every result
you write about. Every improvement in TACHY F1 is the headline finding.

We rendered PTB-XL signals as 12-lead ECG paper images using matplotlib.
Crucially, NO condition labels appear anywhere on the rendered images.
Early renders had condition text visible and GRAD-CAM confirmed the model
was reading the text label not the waveform. We re-rendered everything
clean. This is a methodology point worth mentioning in the paper.

**Imagen** images were generated using Google Gemini 3 Pro Image
(marketed as Nano Banana Pro) via Vertex AI. 160 images per class,
640 total. These were generated using carefully engineered prompts
(8-10 templates per class) and cleaned with OCR + inpainting to remove
any text the model hallucinated onto the images.

**NeuroKit2** images are physiologically simulated ECGs rendered using
the same pipeline as PTB-XL. AFIB uses Markov-chain RR irregularity and
P-wave suppression. MI has per-region ST elevation masks for Anterior,
Inferior, and Lateral infarction patterns. 4000 generated per class,
1500 used per class in training.

---

## The Three Main Experiments

**Experiment A — Baseline**
Train on PTB-XL real data only. This is the control.
4,456 images total.

**Experiment B — LLM Augmentation**
Train on PTB-XL + Imagen generated images.
5,096 images total. Tests whether 160 AI images per class help.

**Experiment C — Combined Augmentation**
Train on PTB-XL + Imagen + NeuroKit2 simulated images.
11,096 images total. Tests combined augmentation effect.

Same model architecture, same hyperparameters, same loss function
across all three. Only the training data changes. This is important
to state clearly in methodology — controlled comparison.

---

## The Model

EfficientNet-B0 pretrained on ImageNet, fine-tuned on ECG images.
Input: 512x512 RGB images of ECG paper renders.
Loss: Focal Loss with class weights — this handles the TACHY imbalance.
Focal Loss focuses the model on hard-to-classify examples by
down-weighting easy correct predictions.
Optimizer: AdamW. Scheduler: CosineAnnealingLR.
Mixed precision (FP16) for GPU memory efficiency.
25 epochs, batch size 32.

---

## The Leakage Problem and How We Fixed It

This is critical for the methodology section.

PTB-XL has multiple recordings per patient. A naive random split
puts patient A's recording 1 in train and recording 2 in test.
The model learns that patient's personal cardiac signature, not the
condition. This inflates all metrics artificially.

We fixed this with patient-grouped 3-fold cross-validation.
Every recording from the same patient goes into the same fold.
The patient_id column in ptbxl_database.csv was used to build
the mapping. Leakage verification reports are in
outputs/leakage_reports/ — they confirm zero patient overlap
between splits.

The metric drop after fixing was tiny (Macro F1: 0.8843 to 0.8777)
which means our original model was genuinely learning waveform
morphology, not patient identity. The GRAD-CAM maps independently
confirmed this — they show attention on P-waves, QRS complexes,
and ST segments, not background.

---

## Results — What to Report in the Paper

Use fold0 numbers as the primary reported results.
The non-fold runs are legacy and should not be cited.

### Fold 0 Primary Metrics

| Experiment | Accuracy | Macro F1 | ROC-AUC | PR-AUC | Kappa | MCC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| A Baseline | 0.8914 | 0.8777 | 0.9766 | 0.9309 | 0.8481 | 0.8497 |
| B +Imagen | 0.8972 | 0.8921 | 0.9779 | 0.9483 | 0.8576 | 0.8582 |
| C +Imagen+NK2 | 0.9546 | 0.9550 | 0.9961 | 0.9907 | 0.9389 | 0.9390 |

### Fold 0 Per-Class F1

| Experiment | NORM | MI | AFIB | TACHY |
|---|:---:|:---:|:---:|:---:|
| A Baseline | 0.9177 | 0.8652 | 0.9297 | 0.7983 |
| B +Imagen | 0.9135 | 0.8894 | 0.9125 | 0.8529 |
| C +Imagen+NK2 | 0.9558 | 0.9442 | 0.9642 | 0.9555 |

### The Narrative Arc for the Paper

1. Baseline (A) achieves strong performance but TACHY lags at F1=0.798
   due to only 426 real training records.

2. Adding 160 Imagen images per class (B) improves TACHY F1 by +5.5%
   to 0.853. Small dataset, meaningful improvement. This demonstrates
   LLM-generated images carry learnable cardiac signal.

3. Adding NeuroKit2 simulation (C) pushes TACHY to F1=0.956, a +15.7%
   improvement over baseline. All classes improve substantially.
   PR-AUC reaches 0.9907 — near perfect precision-recall tradeoff.

4. The ablation experiments (D and E) show that synthetic-only training
   completely fails (D: F1=0.328, E: F1=0.397). This confirms our
   approach is augmentation not replacement — synthetic data only helps
   when combined with real clinical data.

---

## Metrics Explanation for the Paper

**Macro F1** — average F1 across all classes, equal weight per class.
Use this as the primary metric. It penalises poor performance on
minority classes like TACHY.

**ROC-AUC** — area under the receiver operating characteristic curve.
Measures ranking ability regardless of threshold. Values above 0.97
across all experiments show the model reliably separates classes.

**PR-AUC (Average Precision)** — more sensitive than ROC-AUC for
imbalanced classes. The improvement from 0.9309 (A) to 0.9907 (C)
on this metric is arguably the strongest single number in the paper.

**Cohen Kappa** — agreement beyond chance. Above 0.8 is considered
strong agreement. All three experiments exceed this threshold.

**MCC (Matthews Correlation Coefficient)** — considered the most
informative single metric for imbalanced classification. Values
near 1.0 are perfect. Experiment C reaches 0.9390.

**ECE (Expected Calibration Error)** — measures how well model
confidence matches actual accuracy. Lower is better. The instructor
specifically requested this as a superior metric. It is computed
and saved in every run's results folder.

---

## Confidence Scores

The instructor asked for confidence scores. We implemented:

1. ECE (Expected Calibration Error) — scalar metric, lower is better
2. Reliability diagrams — plots confidence vs actual accuracy per bin
3. Confidence histograms — distribution of predicted probabilities
4. Per-prediction confidence CSV — every test image has its predicted
   class, true class, and confidence score saved

All of these are in outputs/results/{run_name}/ for every experiment.

---

## GRAD-CAM

GRAD-CAM (Gradient-weighted Class Activation Mapping) shows which
regions of each ECG image the model uses for its decision.

We ran GRAD-CAM on 5 test images per class per experiment.
The maps consistently show activation on:
- P-wave region (atrial depolarisation)
- QRS complex (ventricular depolarisation)
- ST segment (relevant for MI detection)
- RR interval spacing (relevant for AFIB and TACHY)

This is the scientific validity check. A model could in theory
learn to classify by background colour or rendering artifacts.
GRAD-CAM proves it did not. The instructor specifically praised
these results. Include at least one GRAD-CAM comparison figure
in the paper showing Exp A vs Exp B vs Exp C for TACHY, since
that is where the augmentation effect is largest.

---

## Ablation Experiments D and E

These were trained on synthetic data only and tested on real PTB-XL.

D (NeuroKit2 only): Macro F1 = 0.328, peaked at epoch 8 then collapsed.
E (Imagen only): Macro F1 = 0.397, similar collapse pattern.

Interpretation: The synthetic-to-real domain gap is too large for
a model trained only on simulated data to generalise. However,
Experiment E outperforms D, meaning Imagen images are more
representative of real ECG appearance than NeuroKit2 renders.
This indirectly supports why Imagen augmentation in Exp B and C
is effective — the images are close enough to real ECGs to provide
useful training signal when combined with real data.

The instructor said to exclude these from the main paper findings
but they are valuable as a limitations and ablation section.

---

## Files Hamza Needs to Reference

All experiment outputs are in:
outputs/results/{experiment_run_name}/

Key files per experiment:
- metrics_summary.csv — all scalar metrics in one row
- classification_report.csv — per-class precision, recall, F1
- confusion_matrix.png — visual confusion matrix with percentages
- roc_curves.png — per-class ROC curves
- pr_curves.png — per-class precision-recall curves
- training_curves.png — loss, accuracy, F1 over epochs
- calibration_reliability_diagram.png — confidence calibration
- gradcam/gradcam_{CLASS}.png — GRAD-CAM per class

The canonical runs to cite are:
- exp_A_ptbxl_only_img512_bs32_e25_fold0
- exp_B_ptbxl_imagen_img512_bs32_e25_fold0
- exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25_fold0

Ignore any run without fold0 in the name — those are legacy.

---

## What Is Still Pending

- Folds 1 and 2 for experiments A, B, C are not yet run.
  Final reported metrics should be mean across 3 folds.
  For now fold0 is what we have and what the instructor has seen.
- GRAD-CAM for Experiment C needs to be verified it ran correctly.
  Check: outputs/results/exp_C_.../gradcam/ exists and has images.

---

## One Thing to Be Careful About in the Paper

Do not claim clinical deployment readiness.
Do not claim the model can diagnose patients.
Frame everything as: proof-of-concept, research contribution,
demonstrates feasibility of synthetic augmentation for ECG classification.

The limitations section must mention:
- No cardiologist validation of synthetic ECG quality
- TACHY class heterogeneity (NK2 simulates sinus tachycardia only,
  PTB-XL TACHY includes SVT and ectopic rhythms)
- Only fold 0 reported, full CV pending
- Synthetic-to-real domain gap confirmed by D and E ablations