"""
CV utilities: GroupKFold splitting with patient-level grouping and leakage verification.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


def extract_ecg_id(filepath):
    """Extract ecg_id from PTB-XL filepath (e.g., '.../NORM/12743.png' -> 12743)"""
    filename = os.path.basename(filepath)
    import re
    match = re.search(r'(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    return None


def load_patient_mapping(mapping_csv_path):
    """Load PTB-XL patient_id mapping."""
    df = pd.read_csv(mapping_csv_path)
    # Create lookup: ecg_id -> patient_id
    ecg_to_patient = dict(zip(df['ecg_id'], df['patient_id']))
    return ecg_to_patient


def add_patient_ids(df, ecg_to_patient):
    """Add patient_id column to dataframe based on ecg_id extracted from filepath."""
    df = df.copy()
    df['ecg_id'] = df['filepath'].apply(extract_ecg_id)
    df['patient_id'] = df['ecg_id'].map(ecg_to_patient)
    return df


def create_cv_splits(df, n_splits=3, random_state=42, mapping_csv=None, base_dir=None):
    """
    Create stratified K-fold splits grouped by patient_id (PTB-XL records only).
    Ensures ZERO patient overlap between train/val/test within each fold.

    Strategy:
    1. Use GroupKFold to split PTB-XL into n_splits+1 groups by patient
    2. Use first n_splits groups for train/val/test
    3. Split patients (not records) stratified by majority class

    Returns:
        List of tuples: [(train_idx, val_idx, test_idx), ...]
    """
    if base_dir is None:
        base_dir = Path.cwd()

    if mapping_csv is None:
        mapping_csv = base_dir / "outputs" / "ptbxl_image_patient_mapping.csv"

    # Load mapping
    ecg_to_patient = load_patient_mapping(mapping_csv)

    # Add patient_id to dataframe (for PTB-XL records only)
    df = df.copy()
    ptbxl_mask = df['source'] == 'ptbxl'
    df.loc[ptbxl_mask, 'patient_id'] = df.loc[ptbxl_mask, 'filepath'].apply(extract_ecg_id).map(ecg_to_patient)

    # For synthetic records, assign fake patient_ids (no grouping needed)
    df.loc[~ptbxl_mask, 'patient_id'] = -df.loc[~ptbxl_mask].index - 1  # negative IDs, unique

    # Ensure patient_id is numeric
    df['patient_id'] = pd.to_numeric(df['patient_id'], errors='coerce').fillna(-1).astype(int)

    # Encode labels for stratification
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    groups = df['patient_id'].values
    y = df['label_encoded'].values

    # Use GroupKFold with n_splits+1 to get n_splits valid folds
    gkf = GroupKFold(n_splits=n_splits + 1)

    splits = []
    fold_idx = 0
    for train_val_idx, test_idx in gkf.split(df, y, groups):
        if fold_idx >= n_splits:
            break

        # Get PTB-XL subset for train_val
        train_val_df = df.iloc[train_val_idx].copy()

        # Get unique patients in train_val
        train_val_patients = train_val_df['patient_id'].unique()

        # Compute majority class for each patient in train_val
        patient_class_map = {}
        for pid in train_val_patients:
            patient_records = train_val_df[train_val_df['patient_id'] == pid]
            majority_class = patient_records['label_encoded'].mode()[0]
            patient_class_map[pid] = majority_class

        # Split patients (not records) into train/val groups
        from sklearn.model_selection import train_test_split as sklearn_train_test_split
        patient_classes = np.array([patient_class_map[pid] for pid in train_val_patients])

        train_patients, val_patients = sklearn_train_test_split(
            train_val_patients,
            test_size=0.15,  # ~15% patients for val
            stratify=patient_classes,
            random_state=random_state + fold_idx
        )

        # Map patients back to record indices
        train_idx_final = train_val_df[train_val_df['patient_id'].isin(train_patients)].index.tolist()
        val_idx_final = train_val_df[train_val_df['patient_id'].isin(val_patients)].index.tolist()

        # Convert to positional indices for compatibility
        all_idx = np.arange(len(df))
        train_idx_final = np.where(np.isin(all_idx, train_idx_final))[0]
        val_idx_final = np.where(np.isin(all_idx, val_idx_final))[0]
        test_idx_final = np.where(np.isin(all_idx, test_idx))[0]

        splits.append((train_idx_final, val_idx_final, test_idx_final))
        fold_idx += 1

    return splits


def get_fold_data(df, fold_splits, fold_num):
    """Get train/val/test DataFrames for a specific fold."""
    train_idx, val_idx, test_idx = fold_splits[fold_num]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def verify_no_leakage(train_df, val_df, test_df, fold_num=None, base_dir=None):
    """
    Verify no patient-level leakage between splits.
    Print and return leakage report.
    """
    if base_dir is None:
        base_dir = Path.cwd()

    # Extract patient IDs from PTB-XL records only
    def get_ptbxl_patients(df):
        ptbxl_df = df[df['source'] == 'ptbxl'].copy()
        if len(ptbxl_df) == 0:
            return set()
        mapping_csv = base_dir / "outputs" / "ptbxl_image_patient_mapping.csv"
        ecg_to_patient = load_patient_mapping(mapping_csv)
        ptbxl_df['ecg_id'] = ptbxl_df['filepath'].apply(extract_ecg_id)
        ptbxl_df['patient_id'] = ptbxl_df['ecg_id'].map(ecg_to_patient)
        return set(ptbxl_df['patient_id'].dropna().astype(int).unique())

    train_patients = get_ptbxl_patients(train_df)
    val_patients = get_ptbxl_patients(val_df)
    test_patients = get_ptbxl_patients(test_df)

    # Check overlaps
    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients

    # Generate report
    report = f"""
LEAKAGE VERIFICATION REPORT
{'='*60}
Fold: {fold_num if fold_num is not None else 'N/A'}

Train PTB-XL records: {len(train_df[train_df['source'] == 'ptbxl'])} | Unique patients: {len(train_patients)}
Val   PTB-XL records: {len(val_df[val_df['source'] == 'ptbxl'])} | Unique patients: {len(val_patients)}
Test  PTB-XL records: {len(test_df[test_df['source'] == 'ptbxl'])} | Unique patients: {len(test_patients)}

Patient Overlap:
  Train x Val:  {len(train_val_overlap)} patients ({100*len(train_val_overlap)/max(len(train_patients),1):.1f}% of train)
  Train x Test: {len(train_test_overlap)} patients ({100*len(train_test_overlap)/max(len(train_patients),1):.1f}% of train)
  Val x Test:   {len(val_test_overlap)} patients ({100*len(val_test_overlap)/max(len(val_patients),1):.1f}% of val)

Leakage Status: {'[PASSED] No patient overlap' if not (train_val_overlap or train_test_overlap or val_test_overlap) else '[FAILED] Patient overlap detected!'}
{'='*60}
"""

    print(report)

    # Save report if fold_num specified
    if fold_num is not None:
        report_dir = base_dir / "outputs" / "leakage_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / f"leakage_verification_fold{fold_num}.txt", "w") as f:
            f.write(report)

    return report, bool(train_val_overlap or train_test_overlap or val_test_overlap)


def print_fold_class_distribution(train_df, val_df, test_df, fold_num=None):
    """Print per-class distribution for a fold."""
    classes = ["NORM", "MI", "AFIB", "TACHY"]

    print(f"\n{'='*60}")
    print(f"FOLD {fold_num} — CLASS DISTRIBUTION" if fold_num is not None else "CLASS DISTRIBUTION")
    print('='*60)

    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{split_name}:")
        for cls in classes:
            count = len(split_df[split_df['label'] == cls])
            pct = 100 * count / len(split_df)
            print(f"  {cls:6s}: {count:5d} ({pct:5.1f}%)")
