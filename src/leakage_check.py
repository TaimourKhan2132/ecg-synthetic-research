import pandas as pd
import os
import re
from pathlib import Path

# Paths
PTBXL_DB = "data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/ptbxl_database.csv"
RENDERED_META = "metadata/ptbxl_rendered.csv"
SPLITS_DIR = "data/splits"

# Load PTB-XL database
ptbxl_db = pd.read_csv(PTBXL_DB)
ecg_to_patient = dict(zip(ptbxl_db['ecg_id'], ptbxl_db['patient_id']))

print(f"Total PTB-XL records: {len(ptbxl_db)}")
print(f"Unique patients: {ptbxl_db['patient_id'].nunique()}")
print(f"Avg recordings per patient: {len(ptbxl_db) / ptbxl_db['patient_id'].nunique():.2f}\n")

def extract_ecg_id(filepath):
    """Extract ecg_id from PTB-XL filepath (e.g., '.../NORM/12743.png' -> 12743)"""
    filename = os.path.basename(filepath)
    # Match numeric filename (e.g., 12743.png, 03194.png)
    match = re.search(r'(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    return None

def load_split_with_patient_ids(split_csv):
    """Load split CSV and add patient_id for PTB-XL records"""
    df = pd.read_csv(split_csv)
    df['ecg_id'] = df['filepath'].apply(extract_ecg_id)
    df['patient_id'] = df['ecg_id'].map(ecg_to_patient)
    return df

# Analyze each experiment
for exp_dir in sorted(os.listdir(SPLITS_DIR)):
    exp_path = os.path.join(SPLITS_DIR, exp_dir)
    if not os.path.isdir(exp_path):
        continue

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_dir}")
    print('='*70)

    train_csv = os.path.join(exp_path, "train.csv")
    val_csv = os.path.join(exp_path, "val.csv")
    test_csv = os.path.join(exp_path, "test.csv")

    if not all(os.path.exists(f) for f in [train_csv, val_csv, test_csv]):
        print("  (missing splits)")
        continue

    train = load_split_with_patient_ids(train_csv)
    val = load_split_with_patient_ids(val_csv)
    test = load_split_with_patient_ids(test_csv)

    # Only analyze PTB-XL records
    train_ptbxl = train[train['source'] == 'ptbxl'].dropna(subset=['patient_id'])
    val_ptbxl = val[val['source'] == 'ptbxl'].dropna(subset=['patient_id'])
    test_ptbxl = test[test['source'] == 'ptbxl'].dropna(subset=['patient_id'])

    train_patients = set(train_ptbxl['patient_id'].astype(int).unique())
    val_patients = set(val_ptbxl['patient_id'].astype(int).unique())
    test_patients = set(test_ptbxl['patient_id'].astype(int).unique())

    print(f"Train PTB-XL records: {len(train_ptbxl)} | Unique patients: {len(train_patients)}")
    print(f"Val   PTB-XL records: {len(val_ptbxl)} | Unique patients: {len(val_patients)}")
    print(f"Test  PTB-XL records: {len(test_ptbxl)} | Unique patients: {len(test_patients)}")

    # Check overlaps
    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients

    print(f"\nPatient Leakage:")
    print(f"  Train x Val:  {len(train_val_overlap)} patients ({len(train_val_overlap)*100/len(train_patients):.1f}% of train)")
    print(f"  Train x Test: {len(train_test_overlap)} patients ({len(train_test_overlap)*100/len(train_patients):.1f}% of train)")
    print(f"  Val x Test:   {len(val_test_overlap)} patients ({len(val_test_overlap)*100/len(val_patients):.1f}% of val)")

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("  [!] LEAKAGE DETECTED!")
    else:
        print("  [OK] No patient-level leakage")

    # Show some stats
    total_split_patients = len(train_patients | val_patients | test_patients)
    print(f"\nTotal unique patients across splits: {total_split_patients}")
