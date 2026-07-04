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


def create_cv_splits(df, n_splits=3, random_state=42, mapping_csv=None,
                     base_dir=None, shuffle=False):
    """
    Patient-grouped K-fold splits with a REAL-ONLY validation/test protocol.

    The CV partition is computed over real PTB-XL patients only, so:
      * val and test contain ONLY real PTB-XL records (never synthetic),
      * all synthetic images (Imagen, NeuroKit2) go into TRAIN only,
      * the real val/test folds are IDENTICAL across experiments A/B/C
        (they depend only on the real patients + seed), giving a true
        single-variable comparison where only the training data changes.

    Every real record is tested exactly once (test folds partition the real
    set). Positional indices are into the passed df.

    Returns:
        List of tuples: [(train_idx, val_idx, test_idx), ...]
    """
    from sklearn.model_selection import train_test_split as sklearn_train_test_split

    if base_dir is None:
        base_dir = Path.cwd()
    if mapping_csv is None:
        mapping_csv = base_dir / "outputs" / "ptbxl_image_patient_mapping.csv"

    ecg_to_patient = load_patient_mapping(mapping_csv)

    df = df.copy().reset_index(drop=True)   # positional index == row number

    # --- Real PTB-XL records with a known patient_id -------------------------
    real_df = df[df['source'] == 'ptbxl'].copy()
    real_df['patient_id'] = real_df['filepath'].apply(extract_ecg_id).map(ecg_to_patient)
    real_df = real_df.dropna(subset=['patient_id'])
    real_df['patient_id'] = real_df['patient_id'].astype(int)

    le = LabelEncoder()
    real_df['label_encoded'] = le.fit_transform(real_df['label'])

    # Everything that is NOT a mapped real record (all synthetic + any unmapped
    # real) always goes into TRAIN so no image is ever dropped.
    real_positions = set(real_df.index.values)
    always_train = np.array(sorted(set(range(len(df))) - real_positions), dtype=int)

    groups = real_df['patient_id'].values
    y = real_df['label_encoded'].values

    if shuffle:
        # Repeated CV: seeded, stratified, still patient-grouped -> different
        # partitions per seed (GroupKFold alone is deterministic/seed-invariant).
        from sklearn.model_selection import StratifiedGroupKFold
        gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                   random_state=random_state)
    else:
        gkf = GroupKFold(n_splits=n_splits)

    splits = []
    fold_idx = 0
    for tv_local, test_local in gkf.split(real_df, y, groups):
        tv_df   = real_df.iloc[tv_local]
        test_pos = real_df.iloc[test_local].index.values   # real-only test

        tv_patients = tv_df['patient_id'].unique()
        patient_class_map = {
            pid: tv_df[tv_df['patient_id'] == pid]['label_encoded'].mode()[0]
            for pid in tv_patients
        }
        patient_classes = np.array([patient_class_map[p] for p in tv_patients])

        train_patients, val_patients = sklearn_train_test_split(
            tv_patients,
            test_size=0.15,                       # ~15% of patients for val
            stratify=patient_classes,
            random_state=random_state + fold_idx
        )

        train_real_pos = tv_df[tv_df['patient_id'].isin(train_patients)].index.values
        val_pos        = tv_df[tv_df['patient_id'].isin(val_patients)].index.values  # real-only val

        # Synthetic (and any unmapped) records augment TRAIN only.
        train_pos = np.concatenate([train_real_pos, always_train])

        splits.append((np.sort(train_pos), np.sort(val_pos), np.sort(test_pos)))
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
