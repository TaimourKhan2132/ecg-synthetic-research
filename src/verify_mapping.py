import os
import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(r"C:\Users\taimo\OneDrive\Documents 1\work\ecg-synthetic-research")
PTBXL_DB = BASE_DIR / "data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/ptbxl_database.csv"
PTBXL_RENDERED_DIR = BASE_DIR / "data/rendered/ptbxl"

# Load database
db = pd.read_csv(PTBXL_DB)
ecg_id_to_patient = dict(zip(db['ecg_id'], db['patient_id']))

# Scan rendered directory
total = 0
mapped = 0
unmapped = []

for cls_dir in PTBXL_RENDERED_DIR.iterdir():
    if not cls_dir.is_dir():
        continue
    for img_file in cls_dir.glob("*.png"):
        total += 1
        filename = img_file.stem  # e.g., "00001"
        try:
            ecg_id = int(filename)
            if ecg_id in ecg_id_to_patient:
                mapped += 1
            else:
                unmapped.append((filename, "ecg_id not in database"))
        except ValueError:
            unmapped.append((filename, "non-numeric filename"))

print(f"Total PTB-XL images found: {total}")
print(f"Successfully mapped: {mapped} ({100*mapped/total:.1f}%)")
print(f"Unmapped: {len(unmapped)} ({100*len(unmapped)/total:.1f}%)")

if unmapped:
    print(f"\nUnmapped files (first 10):")
    for fn, reason in unmapped[:10]:
        print(f"  {fn}: {reason}")

print(f"\nVerification: Patient ID mapping is {'RELIABLE' if mapped == total else 'INCOMPLETE'}")
