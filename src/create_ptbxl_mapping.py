import pandas as pd

# Load PTB-XL database and rendered metadata
ptbxl_db = pd.read_csv("data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/ptbxl_database.csv")
rendered = pd.read_csv("metadata/ptbxl_rendered.csv")

# Merge to get patient_id for each record
mapping = rendered.merge(
    ptbxl_db[['ecg_id', 'patient_id']],
    on='ecg_id',
    how='left'
)

# Select and rename columns
mapping = mapping[['file_path', 'label', 'ecg_id', 'patient_id']]
mapping.columns = ['image_path', 'label', 'ecg_id', 'patient_id']

# Save
mapping.to_csv("outputs/ptbxl_image_patient_mapping.csv", index=False)
print(f"Saved mapping: {len(mapping)} records")
print(f"\nSample:\n{mapping.head(10)}")
