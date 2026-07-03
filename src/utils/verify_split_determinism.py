# =============================================================================
# verify_split_determinism.py
# Confirms that fold N uses the SAME PTB-XL patient split across Exp A, B, C.
# Required before any paired statistical test across experiments.
# Usage: python src/utils/verify_split_determinism.py
# =============================================================================

import pandas as pd
from pathlib import Path

BASE_DIR   = Path(r"C:\Users\taimo\OneDrive\Documents 1\work\ecg-synthetic-research")
SPLITS_DIR = BASE_DIR / "data" / "splits"
MAPPING    = BASE_DIR / "outputs" / "ptbxl_image_patient_mapping.csv"

RUN_NAMES = {
    "A": "exp_A_ptbxl_only_img512_bs32_e25",
    "B": "exp_B_ptbxl_imagen_img512_bs32_e25",
    "C": "exp_C_ptbxl_imagen_neurokit2_img512_bs32_e25",
}

mapping_df = pd.read_csv(MAPPING)
# image_path -> patient_id lookup
path_to_patient = dict(zip(mapping_df["image_path"], mapping_df["patient_id"]))


def get_test_patients(exp, fold):
    run_name = f"{RUN_NAMES[exp]}_fold{fold}"
    test_csv = SPLITS_DIR / run_name / "test.csv"
    if not test_csv.exists():
        print(f"[MISSING] {test_csv}")
        return None
    df = pd.read_csv(test_csv)
    # Only PTB-XL rows (source == 'ptbxl') will be in the mapping;
    # Imagen/NeuroKit2 rows have no patient_id and are skipped.
    ptbxl_rows = df[df["source"] == "ptbxl"]
    patients = set()
    matched, unmatched = 0, 0
    for fp in ptbxl_rows["filepath"]:
        pid = path_to_patient.get(fp)
        if pid is not None:
            patients.add(pid)
            matched += 1
        else:
            unmatched += 1
    if unmatched > 0:
        print(f"    [NOTE] Exp {exp} fold{fold}: {unmatched} ptbxl filepaths "
              f"not found in mapping (matched {matched})")
    return patients


for fold in [0, 1, 2]:
    print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")
    patient_sets = {}
    for exp in ["A", "B", "C"]:
        pts = get_test_patients(exp, fold)
        patient_sets[exp] = pts
        print(f"  Exp {exp}: {len(pts) if pts is not None else 'N/A'} PTB-XL test patients")

    if all(v is not None for v in patient_sets.values()):
        ab_match = patient_sets["A"] == patient_sets["B"]
        ac_match = patient_sets["A"] == patient_sets["C"]
        bc_match = patient_sets["B"] == patient_sets["C"]
        print(f"  A == B patient sets? {ab_match}")
        print(f"  A == C patient sets? {ac_match}")
        print(f"  B == C patient sets? {bc_match}")
        if not (ab_match and ac_match and bc_match):
            print(f"  [WARNING] Fold {fold} splits DIFFER across experiments!")
            diff_ab = patient_sets['A'].symmetric_difference(patient_sets['B'])
            print(f"    A-B symmetric diff count: {len(diff_ab)}")