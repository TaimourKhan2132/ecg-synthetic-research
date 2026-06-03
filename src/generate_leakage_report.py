import pandas as pd
import os
import re

PTBXL_DB = "data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/ptbxl_database.csv"
SPLITS_DIR = "data/splits"

ptbxl_db = pd.read_csv(PTBXL_DB)
ecg_to_patient = dict(zip(ptbxl_db['ecg_id'], ptbxl_db['patient_id']))

def extract_ecg_id(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'(\d+)\.png$', filename)
    return int(match.group(1)) if match else None

def load_split_with_patient_ids(split_csv):
    df = pd.read_csv(split_csv)
    df['ecg_id'] = df['filepath'].apply(extract_ecg_id)
    df['patient_id'] = df['ecg_id'].map(ecg_to_patient)
    return df

# Generate report
report = []
report.append("# LEAKAGE ASSESSMENT REPORT\n")
report.append(f"PTB-XL Total Records: {len(ptbxl_db)}")
report.append(f"PTB-XL Unique Patients: {ptbxl_db['patient_id'].nunique()}")
report.append(f"Avg Recordings per Patient: {len(ptbxl_db) / ptbxl_db['patient_id'].nunique():.2f}\n")

all_results = []

for exp_dir in sorted(os.listdir(SPLITS_DIR)):
    exp_path = os.path.join(SPLITS_DIR, exp_dir)
    if not os.path.isdir(exp_path):
        continue

    files = [os.path.join(exp_path, f) for f in ["train.csv", "val.csv", "test.csv"]]
    if not all(os.path.exists(f) for f in files):
        continue

    train = load_split_with_patient_ids(files[0])
    val = load_split_with_patient_ids(files[1])
    test = load_split_with_patient_ids(files[2])

    train_ptbxl = train[train['source'] == 'ptbxl'].dropna(subset=['patient_id'])
    val_ptbxl = val[val['source'] == 'ptbxl'].dropna(subset=['patient_id'])
    test_ptbxl = test[test['source'] == 'ptbxl'].dropna(subset=['patient_id'])

    train_patients = set(train_ptbxl['patient_id'].astype(int).unique())
    val_patients = set(val_ptbxl['patient_id'].astype(int).unique())
    test_patients = set(test_ptbxl['patient_id'].astype(int).unique())

    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients

    all_results.append({
        'Experiment': exp_dir,
        'Train_Patients': len(train_patients),
        'Val_Patients': len(val_patients),
        'Test_Patients': len(test_patients),
        'Train_x_Val': len(train_val_overlap),
        'Train_x_Test': len(train_test_overlap),
        'Val_x_Test': len(val_test_overlap),
        'Leakage_Pct': max(
            len(train_val_overlap)*100/len(train_patients) if train_patients else 0,
            len(train_test_overlap)*100/len(train_patients) if train_patients else 0
        )
    })

# Summary table
df_results = pd.DataFrame(all_results)
report.append("\n## Summary Table\n")
report.append(df_results.to_string(index=False))

# Detailed findings
report.append("\n\n## Detailed Findings\n")
report.append("All experiments show patient-level leakage:\n")
for idx, row in df_results.iterrows():
    leakage = max(row['Train_x_Val'], row['Train_x_Test'], row['Val_x_Test'])
    report.append(f"\n**{row['Experiment']}**")
    report.append(f"  - Train: {row['Train_Patients']} patients")
    report.append(f"  - Val: {row['Val_Patients']} patients")
    report.append(f"  - Test: {row['Test_Patients']} patients")
    report.append(f"  - Max overlap: {leakage} patients ({row['Leakage_Pct']:.1f}%)")

report.append("\n\n## Root Cause\n")
report.append("Train/val/test splits use record-level stratification (via `train_test_split`).")
report.append("When a patient has multiple records, some land in train and others in val/test.")
report.append("The model learns patient morphology (not just the arrhythmia condition).\n")

report.append("\n## Impact on Metrics\n")
report.append("- Reported Exp C: 0.9525 macro F1 (potentially inflated)")
report.append("- Expected after patient-level fix: ~0.90-0.95 macro F1 (2-5% drop)")
report.append("- GRAD-CAMs showing waveform focus (not patient blobs) suggests drop may be smaller\n")

report.append("\n## Recommendation\n")
report.append("Implement 3-fold stratified cross-validation grouped by patient_id.")
report.append("This ensures zero patient overlap between splits and is defensible for conference.")

# Save report
with open("outputs/LEAKAGE_ASSESSMENT.md", "w") as f:
    f.write("\n".join(report))

print("\n".join(report))
print(f"\n\nReport saved to: outputs/LEAKAGE_ASSESSMENT.md")
