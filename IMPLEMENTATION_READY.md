# IMPLEMENTATION COMPLETE — Quick Start Guide

## Status: Ready to Train

All utilities created and tested. CV splits verified with ZERO patient leakage.

---

## Running Experiment A (Validation Gate)

### Command (Run 3 times, once per fold):
```bash
# Fold 0
python src/training/train.py --experiment A --fold 0

# Fold 1
python src/training/train.py --experiment A --fold 1

# Fold 2
python src/training/train.py --experiment A --fold 2
```

**Estimated time per fold:** ~40 minutes  
**Total time:** ~2 hours (sequential)  
**GPU memory:** ~6 GB (same as before)

---

## Output Structure

Each fold creates:
```
outputs/results/exp_A_ptbxl_only_img512_bs32_e25_fold{0,1,2}/
├── metrics_summary.csv                         # Main metrics
├── calibration_reliability_diagram.png         # NEW: Calibration plot
├── calibration_confidence_histogram.png        # NEW: Confidence distribution
├── predictions_with_confidence.csv             # NEW: Per-sample confidence
├── confusion_matrix.png
├── training_curves.png
└── [other existing files]

outputs/leakage_reports/
├── leakage_verification_fold0.txt              # NEW: Zero-leakage proof
├── leakage_verification_fold1.txt
└── leakage_verification_fold2.txt
```

---

## What Changed

### Files Modified:
- `src/training/train.py` — Added `--fold` arg, CV support, calibration metrics

### Files Created:
- `src/utils/__init__.py`
- `src/utils/cv_utils.py` — GroupKFold logic with patient-level grouping
- `src/utils/calibration_metrics.py` — ECE, reliability diagrams, confidence logging
- `outputs/ptbxl_image_patient_mapping.csv` — Patient ID lookup (already generated)

### Files Unchanged:
- Rendering, generation, GRAD-CAM — all compatible ✓
- Existing experiment directories — preserved for comparison ✓

---

## Validation Checkpoint

After Exp A (fold 0, 1, 2):
1. Compare mean Macro F1 across folds
2. Check calibration plots and ECE scores
3. Review leakage reports (should say "PASSED" for all)
4. Decide: proceed to B & C if metrics acceptable

---

## Backward Compatibility

**Legacy mode still works:**
```bash
python src/training/train.py --experiment A  # Uses old single split (no --fold)
```

---

## Next Steps

1. **Run Fold 0:** `python src/training/train.py --experiment A --fold 0`
2. Monitor GPU, check outputs
3. If successful, run Folds 1 & 2
4. Aggregate results and compare to old (0.8843)
5. Decide on Experiments B & C
