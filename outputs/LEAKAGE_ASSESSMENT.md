# LEAKAGE ASSESSMENT REPORT

PTB-XL Total Records: 21837
PTB-XL Unique Patients: 18885
Avg Recordings per Patient: 1.16


## Summary Table

                                  Experiment  Train_Patients  Val_Patients  Test_Patients  Train_x_Val  Train_x_Test  Val_x_Test  Leakage_Pct
    exp_A_baseline_img224_bs32_e20_0531_2035            3336           443            440           51            48          14     1.528777
            exp_A_ptbxl_only_img512_bs32_e25            3336           443            440           51            48          14     1.528777
      exp_B_imagen_img224_bs48_e20_0531_2006            3346           444            427           50            47          10     1.494322
          exp_B_ptbxl_imagen_img512_bs32_e25            3346           444            427           50            47          10     1.494322
exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25            3367           428            423           56            46           6     1.663202


## Detailed Findings

All experiments show patient-level leakage:


**exp_A_baseline_img224_bs32_e20_0531_2035**
  - Train: 3336 patients
  - Val: 443 patients
  - Test: 440 patients
  - Max overlap: 51 patients (1.5%)

**exp_A_ptbxl_only_img512_bs32_e25**
  - Train: 3336 patients
  - Val: 443 patients
  - Test: 440 patients
  - Max overlap: 51 patients (1.5%)

**exp_B_imagen_img224_bs48_e20_0531_2006**
  - Train: 3346 patients
  - Val: 444 patients
  - Test: 427 patients
  - Max overlap: 50 patients (1.5%)

**exp_B_ptbxl_imagen_img512_bs32_e25**
  - Train: 3346 patients
  - Val: 444 patients
  - Test: 427 patients
  - Max overlap: 50 patients (1.5%)

**exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25**
  - Train: 3367 patients
  - Val: 428 patients
  - Test: 423 patients
  - Max overlap: 56 patients (1.7%)


## Root Cause

Train/val/test splits use record-level stratification (via `train_test_split`).
When a patient has multiple records, some land in train and others in val/test.
The model learns patient morphology (not just the arrhythmia condition).


## Impact on Metrics

- Reported Exp C: 0.9525 macro F1 (potentially inflated)
- Expected after patient-level fix: ~0.90-0.95 macro F1 (2-5% drop)
- GRAD-CAMs showing waveform focus (not patient blobs) suggests drop may be smaller


## Recommendation

Implement 3-fold stratified cross-validation grouped by patient_id.
This ensures zero patient overlap between splits and is defensible for conference.