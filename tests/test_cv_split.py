"""Regression tests for the evaluation protocol in src/utils/cv_utils.py.

The central guarantee of the study: validation and test sets contain only real
PTB-XL records, synthetic images are used for training only, and no patient
appears in more than one split. An earlier version gave each synthetic image its
own fake patient ID, which let synthetic images leak into val/test; these tests
pin the corrected behaviour.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "utils"))
from cv_utils import create_cv_splits, get_fold_data  # noqa: E402

CLASSES = ["NORM", "MI", "AFIB", "TACHY"]


@pytest.fixture
def dataset(tmp_path):
    """Small fake dataset: 160 patients with 1-3 recordings each, plus synthetic images."""
    rng = np.random.default_rng(0)
    rows, mapping = [], []
    ecg_id = 1
    for pid in range(160):
        label = CLASSES[pid % len(CLASSES)]
        for _ in range(rng.integers(1, 4)):
            rows.append({"filepath": f"data/rendered/ptbxl/{label}/{ecg_id:05d}.png",
                         "label": label, "source": "ptbxl"})
            mapping.append({"ecg_id": ecg_id, "patient_id": 10_000 + pid})
            ecg_id += 1
    real = pd.DataFrame(rows)
    synth_rows = []
    for label in CLASSES:
        for i in range(50):
            src = "imagen" if i % 2 else "neurokit2"
            # digits in the filename on purpose: they must not be mistaken for ecg_ids
            synth_rows.append({"filepath": f"data/rendered/{src}/{label}/{src}_{label}_{i:05d}.png",
                               "label": label, "source": src})
    synth = pd.DataFrame(synth_rows)
    mapping_csv = tmp_path / "mapping.csv"
    pd.DataFrame(mapping).to_csv(mapping_csv, index=False)
    ecg_to_patient = {m["ecg_id"]: m["patient_id"] for m in mapping}
    return real, synth, mapping_csv, ecg_to_patient


def _patients(df, ecg_to_patient):
    ids = df["filepath"].str.extract(r"(\d+)\.png$")[0].astype(int)
    return set(ids.map(ecg_to_patient))


@pytest.mark.parametrize("shuffle", [False, True])
def test_synthetic_images_only_in_train(dataset, shuffle):
    real, synth, mapping_csv, _ = dataset
    df = pd.concat([real, synth], ignore_index=True)
    splits = create_cv_splits(df, n_splits=3, random_state=42, mapping_csv=mapping_csv, shuffle=shuffle)
    for fold in range(3):
        train, val, test = get_fold_data(df, splits, fold)
        assert (val["source"] == "ptbxl").all()
        assert (test["source"] == "ptbxl").all()
        assert (train["source"] != "ptbxl").sum() == len(synth)


@pytest.mark.parametrize("shuffle", [False, True])
def test_no_patient_in_more_than_one_split(dataset, shuffle):
    real, synth, mapping_csv, ecg_to_patient = dataset
    df = pd.concat([real, synth], ignore_index=True)
    splits = create_cv_splits(df, n_splits=3, random_state=42, mapping_csv=mapping_csv, shuffle=shuffle)
    for fold in range(3):
        train, val, test = get_fold_data(df, splits, fold)
        tr = _patients(train[train["source"] == "ptbxl"], ecg_to_patient)
        va, te = _patients(val, ecg_to_patient), _patients(test, ecg_to_patient)
        assert not (tr & va) and not (tr & te) and not (va & te)


def test_test_folds_partition_the_real_set(dataset):
    real, synth, mapping_csv, _ = dataset
    df = pd.concat([real, synth], ignore_index=True)
    splits = create_cv_splits(df, n_splits=3, random_state=42, mapping_csv=mapping_csv)
    tests = [set(df.iloc[s[2]]["filepath"]) for s in splits]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not (tests[i] & tests[j])
    assert set().union(*tests) == set(real["filepath"])


def test_real_splits_do_not_depend_on_augmentation(dataset):
    """Single-variable design: adding synthetic data must not change the real val/test sets."""
    real, synth, mapping_csv, _ = dataset
    with_synth = pd.concat([real, synth], ignore_index=True)
    s_real = create_cv_splits(real, n_splits=3, random_state=42, mapping_csv=mapping_csv)
    s_aug = create_cv_splits(with_synth, n_splits=3, random_state=42, mapping_csv=mapping_csv)
    for a, b in zip(s_real, s_aug):
        for k in (1, 2):  # val, test
            assert set(real.iloc[a[k]]["filepath"]) == set(with_synth.iloc[b[k]]["filepath"])
