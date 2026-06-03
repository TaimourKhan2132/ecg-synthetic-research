# 🫀 ECG Synthetic Augmentation Research

> **Can AI-generated ECG images improve cardiac arrhythmia classification?**
> A comparative study of EfficientNet-B0 trained on real clinical ECGs,
> augmented with Google Imagen 3 (Nano Banana Pro) generated images
> and NeuroKit2 physiologically simulated images.

---

## Overview

This project extends the **Nano Banana pneumonia synthetic imaging paper**
to the ECG domain. We investigate whether synthetic ECG augmentation
improves CNN classification on real PTB-XL clinical data across three
progressive experiments, with full patient-level cross-validation to
prevent data leakage.

**Research Question:**
> Does Imagen-generated synthetic ECG augmentation improve CNN
> classification performance on real PTB-XL ECG data, compared to
> a real-data-only baseline and a combined augmentation baseline?

---

## Experiments

| ID | Training Data | Test Data | Purpose |
|:---:|---|---|---|
| **A** | PTB-XL real (4,456) | PTB-XL patient-split | Baseline |
| **B** | PTB-XL + Imagen (5,096) | PTB-XL patient-split | LLM augmentation |
| **C** | PTB-XL + Imagen + NeuroKit2 (11,096) | PTB-XL patient-split | Combined augmentation |
| **D*** | NeuroKit2 only (6,000) | Full PTB-XL | Domain transfer ablation |
| **E*** | Imagen only (640) | Full PTB-XL | Domain transfer ablation |

> \*Experiments D and E are ablation studies confirming that
> synthetic-only training cannot replace real clinical data.
> D peaked at Macro F1 = 0.328, E at 0.397 — both collapsing
> after epoch 8. This validates the augmentation strategy in A/B/C
> and is discussed in Limitations.

**Validation:** 3-fold patient-grouped cross-validation ensures all
recordings from the same patient land in the same fold, preventing
patient-identity leakage. Verification reports in `outputs/leakage_reports/`.

---

## Dataset

### PTB-XL — Real Clinical Data
- Source: PhysioNet PTB-XL v1.0.3, 21,837 records
- 4 classes: NORM, MI, AFIB, TACHY
- Rendered as clean 12-lead ECG paper images — no text labels
- GRAD-CAM verified: model attends to waveforms, not rendering style
- Balanced: NORM/MI capped at 1500, AFIB ~1030, TACHY ~426

### Imagen — AI Generated
- Model: Gemini 3 Pro Image (Nano Banana Pro) via Google Vertex AI
- 160 images per class (640 total)
- Automated generation with full seed and prompt logging
- OCR-cleaned with EasyOCR + inpainting to remove text artifacts
- 8–10 prompt templates per class for morphological diversity

### NeuroKit2 — Physiological Simulation
- Condition-specific signal simulation with realistic noise injection
- MI: per-region ST elevation + Q-wave masks (Anterior/Inferior/Lateral)
- AFIB: Markov-chain RR irregularity + P-wave suppression
- Identical rendering pipeline to PTB-XL — no text labels
- 4,000 generated per class, 1,500 used in training

### Class Distribution

| Class | Condition | PTB-XL | Imagen | NeuroKit2 |
|---|---|:---:|:---:|:---:|
| NORM | Normal ECG | 1500 | 160 | 1500 |
| MI | Myocardial Infarction | 1500 | 160 | 1500 |
| AFIB | Atrial Fibrillation | 1030 | 160 | 1500 |
| TACHY | Tachycardia | 426 | 160 | 1500 |

> TACHY has the fewest real records (426).
> Augmentation effect is most pronounced here.

---

## Model & Training

| Component | Choice |
|---|---|
| Architecture | EfficientNet-B0 (ImageNet pretrained) |
| Input | 512 × 512 RGB ECG images |
| Loss | Focal Loss γ=2 + class weights |
| Optimizer | AdamW lr=1e-4 wd=1e-4 |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) |
| Precision | FP16 mixed precision (AMP) |
| Epochs | 25 · Batch size 32 |
| Validation | 3-fold patient-grouped CV |
| GPU | NVIDIA RTX 4050 6GB |

---

## Results

> Fold 0 results shown. Non-fold runs use legacy single split
> (pre-leakage-fix, kept for reference only).
> Canonical results are the fold0 rows.

### Primary Metrics — Fold 0

| Experiment | Accuracy | Macro F1 | ROC-AUC | PR-AUC | Kappa | MCC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** Baseline | 0.8914 | 0.8777 | 0.9766 | 0.9309 | 0.8481 | 0.8497 |
| **B** +Imagen | 0.8972 | 0.8921 | 0.9779 | 0.9483 | 0.8576 | 0.8582 |
| **C** +Imagen+NK2 | **0.9546** | **0.9550** | **0.9961** | **0.9907** | **0.9389** | **0.9390** |

### Per-Class F1 — Fold 0

| Experiment | NORM | MI | AFIB | TACHY | Δ TACHY vs A |
|---|:---:|:---:|:---:|:---:|:---:|
| **A** Baseline | 0.9177 | 0.8652 | 0.9297 | 0.7983 | — |
| **B** +Imagen | 0.9135 | 0.8894 | 0.9125 | 0.8529 | **+0.055** |
| **C** +Imagen+NK2 | **0.9558** | **0.9442** | **0.9642** | **0.9555** | **+0.157** |

### Ablation — Domain Transfer (D & E)

| Experiment | Accuracy | Macro F1 | ROC-AUC | Interpretation |
|---|:---:|:---:|:---:|---|
| **D** NK2→PTB-XL | 0.3500 | 0.3281 | 0.6371 | Sim-to-real gap too large |
| **E** Imagen→PTB-XL | 0.4420 | 0.3967 | 0.7212 | Imagen closer to real than NK2 |

### Key Findings

- **TACHY F1: +5.5% (A→B), +15.7% (A→C)** — largest gains on the
  most data-scarce class. Directly validates the augmentation hypothesis.
- **Experiment C Macro F1: 0.9550** — a +7.7% improvement over the
  baseline using combined augmentation.
- **Experiment E outperforms D** (0.397 vs 0.328) — Imagen images
  are more visually representative of real ECGs than NeuroKit2 renders,
  confirming their value as augmentation.
- **Calibration (ECE)** confirms model confidence is reliable
  — not overfit to training distribution.
- **GRAD-CAM** maps show attention on P-waves, QRS complexes, and
  ST segments — clinically meaningful feature learning confirmed.

---

## Visualizations

### Training Curves

| Exp A | Exp B | Exp C |
|:---:|:---:|:---:|
| ![](outputs/results/exp_A_ptbxl_only_img512_bs32_e25_fold0/training_curves.png) | ![](outputs/results/exp_B_ptbxl_imagen_img512_bs32_e25_fold0/training_curves.png) | ![](outputs/results/exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25_fold0/training_curves.png) |

### Confusion Matrices

| Exp A | Exp B | Exp C |
|:---:|:---:|:---:|
| ![](outputs/results/exp_A_ptbxl_only_img512_bs32_e25_fold0/confusion_matrix.png) | ![](outputs/results/exp_B_ptbxl_imagen_img512_bs32_e25_fold0/confusion_matrix.png) | ![](outputs/results/exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25_fold0/confusion_matrix.png) |

### ROC Curves

| Exp A | Exp B | Exp C |
|:---:|:---:|:---:|
| ![](outputs/results/exp_A_ptbxl_only_img512_bs32_e25_fold0/roc_curves.png) | ![](outputs/results/exp_B_ptbxl_imagen_img512_bs32_e25_fold0/roc_curves.png) | ![](outputs/results/exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25_fold0/roc_curves.png) |

### Calibration — Reliability Diagrams

| Exp A | Exp B | Exp C |
|:---:|:---:|:---:|
| ![](outputs/results/exp_A_ptbxl_only_img512_bs32_e25_fold0/calibration_reliability_diagram.png) | ![](outputs/results/exp_B_ptbxl_imagen_img512_bs32_e25_fold0/calibration_reliability_diagram.png) | ![](outputs/results/exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25_fold0/calibration_reliability_diagram.png) |

### GRAD-CAM Explainability

> Model attends to ECG waveform morphology —
> P-waves, QRS complexes, ST segments — not background or grid.

#### Experiment A — Baseline

| NORM | MI | AFIB | TACHY |
|:---:|:---:|:---:|:---:|
| ![](outputs/results/exp_A_ptbxl_only_img512_bs32_e25_fold0/gradcam/gradcam_NORM.png) | ![](outputs/results/exp_A_ptbxl_only_img512_bs32_e25_fold0/gradcam/gradcam_MI.png) | ![](outputs/results/exp_A_ptbxl_only_img512_bs32_e25_fold0/gradcam/gradcam_AFIB.png) | ![](outputs/results/exp_A_ptbxl_only_img512_bs32_e25_fold0/gradcam/gradcam_TACHY.png) |

#### Experiment B — PTB-XL + Imagen

| NORM | MI | AFIB | TACHY |
|:---:|:---:|:---:|:---:|
| ![](outputs/results/exp_B_ptbxl_imagen_img512_bs32_e25_fold0/gradcam/gradcam_NORM.png) | ![](outputs/results/exp_B_ptbxl_imagen_img512_bs32_e25_fold0/gradcam/gradcam_MI.png) | ![](outputs/results/exp_B_ptbxl_imagen_img512_bs32_e25_fold0/gradcam/gradcam_AFIB.png) | ![](outputs/results/exp_B_ptbxl_imagen_img512_bs32_e25_fold0/gradcam/gradcam_TACHY.png) |

#### Experiment C — PTB-XL + Imagen + NeuroKit2

| NORM | MI | AFIB | TACHY |
|:---:|:---:|:---:|:---:|
| ![](outputs/results/exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25_fold0/gradcam/gradcam_NORM.png) | ![](outputs/results/exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25_fold0/gradcam/gradcam_MI.png) | ![](outputs/results/exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25_fold0/gradcam/gradcam_AFIB.png) | ![](outputs/results/exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25_fold0/gradcam/gradcam_TACHY.png) |
---

## Project Structure

```

ecg-synthetic-research/
│
├── src/
│   ├── rendering/
│   │   ├── render_ptbxl.py          PTB-XL signal → ECG image
│   │   ├── render_neurokit2.py      NeuroKit2 simulation → ECG image
│   │   ├── sanitize_imagen.py       OCR + inpaint Imagen text artifacts
│   │   ├── fix_imagen.py            Edge crop + corner mask
│   │   └── fix_neurokit2.py         Strip header text from NK2 images
│   │
│   ├── generation/
│   │   ├── imagen_generate.py       Vertex AI Imagen batch generation
│   │   └── prompts/prompts.csv      Prompt templates per class
│   │
│   ├── training/
│   │   ├── train.py                 Experiments A B C — 3-fold CV
│   │   └── train_cross_domain.py   Experiments D E — ablation
│   │
│   ├── explainability/
│   │   └── gradcam.py               GRAD-CAM per class + overview
│   │
│   ├── utils/
│   │   ├── calibration_metrics.py   ECE, reliability diagrams
│   │   └── cv_utils.py              Patient-grouped CV split logic
│   │
│   ├── viz/
│   │   └── dashboard.py             Streamlit comparison dashboard
│   │
│   ├── create_ptbxl_mapping.py      Patient ID → image mapping
│   ├── generate_leakage_report.py   Leakage verification
│   ├── leakage_check.py             Leakage detection
│   └── verify_mapping.py            Mapping validation
│
├── outputs/
│   ├── results/{run_name}/
│   │   ├── metrics_summary.csv
│   │   ├── classification_report.csv
│   │   ├── confusion_matrix.csv / .png
│   │   ├── roc_curves.csv / .png
│   │   ├── pr_curves.png
│   │   ├── training_curves.png
│   │   ├── probability_distributions.png
│   │   ├── calibration_reliability_diagram.png
│   │   ├── calibration_confidence_histogram.png
│   │   ├── predictions_with_confidence.csv
│   │   └── gradcam/
│   │       ├── gradcam_overview.png
│   │       └── gradcam_{CLASS}.png
│   │
│   ├── leakage_reports/             Fold-level leakage verification
│   ├── ptbxl_image_patient_mapping.csv
│   └── LEAKAGE_ASSESSMENT.md
│
├── data/
│   ├── metadata/                    Dataset manifest CSVs
│   └── splits/                      Train/val/test per fold
│
├── run_all.py                       Sequential overnight runner
├── requirements.txt
└── README.md


```
---

## Setup

```bash
git clone https://github.com/YOURUSERNAME/ecg-synthetic-research.git
cd ecg-synthetic-research

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# CUDA PyTorch — RTX 4050
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
```

## Reproducing Results

```bash
# 1. Render PTB-XL (requires raw dataset download)
python src/rendering/render_ptbxl.py

# 2. Generate Imagen images (requires Google Cloud credentials)
python src/generation/imagen_generate.py
python src/rendering/sanitize_imagen.py

# 3. Generate NeuroKit2 dataset
python src/rendering/render_neurokit2.py

# 4. Build patient mapping (required before CV training)
python src/create_ptbxl_mapping.py

# 5. Train all experiments
python run_all.py

# 6. Launch comparison dashboard
streamlit run src/viz/dashboard.py
```

---

## Limitations

- Synthetic ECGs not validated by cardiologists
- TACHY: 426 real records — thin statistical base
- Imagen may produce visually plausible but physiologically
  inaccurate waveforms — no morphological ground truth validation
- NeuroKit2 TACHY simulates sinus tachycardia only; PTB-XL TACHY
  includes SVT and ectopic rhythms (heterogeneous class)
- Only Fold 0 reported — full 3-fold CV averaging pending
- D/E ablations confirm synthetic-only models fail on real data
  (domain gap too large without real ECG examples)
- Not intended for clinical deployment

---

## Tech Stack

`PyTorch` · `EfficientNet-B0` · `Vertex AI Imagen 3` · `NeuroKit2`
`PTB-XL` · `EasyOCR` · `pytorch-grad-cam` · `Streamlit`
`Focal Loss` · `Mixed Precision AMP` · `Patient-grouped 3-fold CV`

---

## Authors

<table>
  <tr>
    <td align="center">
      <b>TKR</b><br>
      Full pipeline · All experiments<br>
      Rendering · Training · Analysis<br>
      BS Data Science · UMT Lahore
    </td>
    <td align="center">
      <b>Hamza Chaudhry</b><br>
      Methodology review<br>
      Conference report writing<br>
      BS Data Science · UMT Lahore
    </td>
    <td align="center">
      <b>Waleed Nadeem</b><br>
      BS Data Science<br>
      UMT Lahore
    </td>
    <td align="center">
      <b>Hashaam Ijaz</b><br>
      BS Data Science<br>
      UMT Lahore
    </td>
  </tr>
</table>

> Supervised by faculty, Department of Data Science, UMT Lahore.
> Target venue: International Medical AI Conference, Dubai.

---

## License

For academic research only.
PTB-XL data subject to the
[PhysioNet Credentialed Health Data License](https://physionet.org/content/ptb-xl/view-license/1.0.3/).