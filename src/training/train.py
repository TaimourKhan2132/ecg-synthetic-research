# =============================================================================
# train.py
# Fixed experiments A, B, C. 512px. Focal loss. Class weights only.
# Maximum metrics: F1, ROC-AUC, PR-AUC, Kappa, MCC, confusion matrix.
# Usage: python src/training/train.py --experiment A
# =============================================================================

import os
# Reduce CUDA fragmentation on the 6GB card (must be set before torch imports CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import random
import hashlib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, average_precision_score,
    cohen_kappa_score, matthews_corrcoef,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

# Import custom utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import cv_utils, calibration_metrics

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR     = Path(r"C:\Users\taimo\OneDrive\Documents 1\work\ecg-synthetic-research")
PTBXL_DIR    = BASE_DIR / "data" / "rendered" / "ptbxl"
IMAGEN_DIR   = BASE_DIR / "data" / "rendered" / "imagen_clean"
NEUROKIT_DIR = BASE_DIR / "data" / "rendered" / "neurokit2"
SPLITS_DIR   = BASE_DIR / "data"  / "splits"
MODELS_DIR   = BASE_DIR / "outputs" / "models"
RESULTS_DIR  = BASE_DIR / "outputs" / "results"
# Training reads pre-resized 512px copies from here (full-res originals are kept
# for Grad-CAM). The cache stores exactly what transforms.Resize produces, so
# results are identical — it just avoids decoding 2850x1722 PNGs every epoch.
CACHE_DIR    = BASE_DIR / "data" / "cache_train_512"

CLASSES      = ["NORM", "MI", "AFIB", "TACHY"]
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# FIXED EXPERIMENT CONFIG
# =============================================================================

NK_CAP = 1500  # NeuroKit2 images per class

EXPERIMENTS = {
    "A": {
        "name":    "A_ptbxl_only",
        "sources": {"ptbxl": PTBXL_DIR},
        "desc":    "Baseline — PTB-XL real only",
    },
    "B": {
        "name":    "B_ptbxl_imagen",
        "sources": {"ptbxl": PTBXL_DIR, "imagen": IMAGEN_DIR},
        "desc":    "PTB-XL + Imagen augmentation",
    },
    "C": {
        "name":    "C_ptbxl_imagen_neurokit2",
        "sources": {"ptbxl": PTBXL_DIR, "imagen": IMAGEN_DIR,
                    "neurokit2": NEUROKIT_DIR},
        "desc":    "PTB-XL + Imagen + NeuroKit2 augmentation",
    },
}

# Fixed hyperparameters
IMG_SIZE    = 512
# Effective batch is 32 (paper spec), realised as micro-batch 16 x accumulation 2
# because batch 32 @ 512px needs ~7GB and does not fit the 6GB RTX 4050.
# Optimizer sees gradients over 32 samples; only run naming uses BATCH_SIZE.
MICRO_BATCH = 16
ACCUM_STEPS = 2
BATCH_SIZE  = MICRO_BATCH * ACCUM_STEPS   # 32 effective (used for run_name)
EPOCHS      = 25
LR          = 1e-4
NUM_WORKERS = 2   # cache makes decode trivial, so 2 workers saturate the GPU;
                  # keeping it low avoids Windows commit-charge (paging) blowups
                  # from each spawned worker loading the large CUDA DLLs

# =============================================================================
# DATASET
# =============================================================================

def collect_images(sources: dict, synth_cap: int = None,
                   aug_classes: list = None) -> pd.DataFrame:
    """
    synth_cap:   max synthetic images per class per source (None = default;
                 NeuroKit2 still capped at NK_CAP). Real (ptbxl) is never capped.
    aug_classes: if given, ONLY these classes receive synthetic augmentation;
                 other classes stay real-only. Real (ptbxl) always keeps all classes.
    """
    records = []
    for src_name, src_dir in sources.items():
        src_dir = Path(src_dir)
        is_synth = src_name != "ptbxl"
        for cls in CLASSES:
            # Minority-only augmentation: skip synthetic for non-target classes.
            if is_synth and aug_classes is not None and cls not in aug_classes:
                continue
            cls_dir = src_dir / cls
            if not cls_dir.exists():
                print(f"  [WARN] Missing: {cls_dir}")
                continue
            images = sorted(cls_dir.rglob("*.png"))   # sorted -> deterministic

            if is_synth:
                cap = synth_cap if synth_cap is not None else (
                    NK_CAP if src_name == "neurokit2" else None)
                if cap is not None and len(images) > cap:
                    rng = np.random.RandomState(SEED)
                    images = list(rng.choice(images, cap, replace=False))

            for img_path in images:
                records.append({
                    "filepath": str(img_path),
                    "label":    cls,
                    "source":   src_name,
                })

    df = pd.DataFrame(records)
    return df


def make_splits(df: pd.DataFrame, exp_name: str, fold_num: int = None):
    """
    Create train/val/test splits.

    If fold_num is None: Use legacy single split (backward compatible).
    If fold_num is 0, 1, or 2: Use patient-grouped 3-fold CV.
    """
    if fold_num is None:
        # Legacy mode: single split (old behavior)
        split_dir = SPLITS_DIR / exp_name
        split_dir.mkdir(parents=True, exist_ok=True)

        train_df, temp_df = train_test_split(
            df, test_size=0.2, stratify=df["label"], random_state=SEED
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED
        )

        train_df.to_csv(split_dir / "train.csv", index=False)
        val_df.to_csv(split_dir / "val.csv", index=False)
        test_df.to_csv(split_dir / "test.csv", index=False)

        print(f"\nSplit (LEGACY) — Train:{len(train_df)} Val:{len(val_df)} Test:{len(test_df)}")

    else:
        # CV mode: patient-grouped 3-fold
        split_dir = SPLITS_DIR / exp_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create 3-fold CV splits (grouped by patient_id)
        mapping_csv = BASE_DIR / "outputs" / "ptbxl_image_patient_mapping.csv"
        # Repeated CV for seeds != 42 (StratifiedGroupKFold shuffle) so each seed
        # is a genuinely different partition; canonical seed 42 keeps GroupKFold.
        fold_splits = cv_utils.create_cv_splits(df, n_splits=3, random_state=SEED,
                                               mapping_csv=mapping_csv, base_dir=BASE_DIR,
                                               shuffle=(SEED != 42))

        train_df, val_df, test_df = cv_utils.get_fold_data(df, fold_splits, fold_num)

        # Save splits
        train_df.to_csv(split_dir / "train.csv", index=False)
        val_df.to_csv(split_dir / "val.csv", index=False)
        test_df.to_csv(split_dir / "test.csv", index=False)

        # Verify no leakage
        leakage_report, has_leakage = cv_utils.verify_no_leakage(
            train_df, val_df, test_df, fold_num=fold_num, base_dir=BASE_DIR
        )

        if has_leakage:
            print(f"\n[WARNING] Patient-level leakage detected in fold {fold_num}!")
        else:
            print(f"\n[OK] Fold {fold_num}: No patient-level leakage")

        # Print class distribution
        cv_utils.print_fold_class_distribution(train_df, val_df, test_df, fold_num=fold_num)

    print("Train class distribution:")
    for cls in CLASSES:
        n    = len(train_df[train_df["label"] == cls])
        pct  = 100 * n / len(train_df)
        print(f"  {cls:6s}: {n:5d} ({pct:.1f}%)")

    return train_df, val_df, test_df


def seed_worker(worker_id):
    """Reproducible per-worker RNG (derives from the loader generator seed,
    so it varies correctly with --seed)."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# RESIZED TRAINING CACHE
# Full-res ECG renders are 2850x1722 PNGs; decoding them every epoch starves the
# GPU. We cache a one-time 512px copy (exactly transforms.Resize output) so the
# per-epoch decode is ~30x cheaper. Originals are untouched (Grad-CAM uses them).
# =============================================================================

def cache_path_for(filepath):
    h = hashlib.md5(str(filepath).encode()).hexdigest()
    return CACHE_DIR / f"{h}.png"


def _build_one_cache(filepath):
    cp = cache_path_for(filepath)
    if cp.exists():
        return
    from torchvision import transforms as _T
    img = Image.open(filepath).convert("RGB")
    img = _T.Resize((IMG_SIZE, IMG_SIZE))(img)   # identical to the pipeline's Resize
    img.save(cp, format="PNG")


def build_train_cache(df, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = df["filepath"].unique().tolist()
    todo  = [p for p in paths if not cache_path_for(p).exists()]
    if not todo:
        print(f"[cache] all {len(paths)} images already cached at {IMG_SIZE}px")
        return
    print(f"[cache] building {len(todo)}/{len(paths)} resized {IMG_SIZE}px images...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(tqdm(ex.map(_build_one_cache, todo, chunksize=16), total=len(todo)))


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform, use_weights=False):
        self.df           = df.reset_index(drop=True)
        self.transform    = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        self.use_weights  = use_weights and "sample_weight" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        # Read the pre-resized 512px cache; fall back to the original if missing.
        cp    = cache_path_for(row["filepath"])
        src   = cp if cp.exists() else row["filepath"]
        image = Image.open(src).convert("RGB")
        image = self.transform(image)
        label = self.class_to_idx[row["label"]]
        if self.use_weights:
            return image, label, float(row["sample_weight"])
        return image, label

# =============================================================================
# TRANSFORMS
# =============================================================================

def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=2, translate=(0.02, 0.02),
                                scale=(0.97, 1.03)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf

# =============================================================================
# FOCAL LOSS + CLASS WEIGHTS
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal loss with class weighting.
    Focuses training on hard misclassified examples.
    gamma=2 is standard. alpha handles class imbalance.
    """
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma

    def forward(self, inputs, targets, sample_weight=None):
        ce_loss = F.cross_entropy(inputs, targets,
                                  weight=self.alpha, reduction="none")
        pt      = torch.exp(-ce_loss)
        focal   = (1 - pt) ** self.gamma * ce_loss
        if sample_weight is not None:  # H16: down-weight synthetic samples
            return (focal * sample_weight).sum() / (sample_weight.sum() + 1e-8)
        return focal.mean()


def compute_class_weights(train_df, real_only=False):
    # H1: with real_only, focal alpha reflects the REAL class frequencies even
    # though synthetic images are still trained on — so balanced synthetic no
    # longer neutralizes the up-weighting the scarce classes need.
    if real_only and (train_df["source"] == "ptbxl").any():
        train_df = train_df[train_df["source"] == "ptbxl"]
    class_to_idx  = {c: i for i, c in enumerate(CLASSES)}
    labels        = [class_to_idx[l] for l in train_df["label"]]
    counts        = np.bincount(labels, minlength=len(CLASSES))
    weights       = 1.0 / (counts + 1e-6)
    weights       = weights / weights.sum() * len(CLASSES)
    return torch.FloatTensor(weights).to(DEVICE)

# =============================================================================
# MODEL
# =============================================================================

def build_model(arch="b0"):
    if arch == "convnext_tiny":
        model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # ConvNeXt head: Sequential(LayerNorm2d, Flatten, Linear); keep the
        # norm+flatten, swap the final Linear for dropout + 4-class linear.
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.4), nn.Linear(in_features, len(CLASSES)))
        return model.to(DEVICE)
    if arch == "b1":
        model = models.efficientnet_b1(
            weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, len(CLASSES))
    )
    return model.to(DEVICE)

# =============================================================================
# TRAIN / EVAL LOOPS
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, scaler,
                freeze_bn=False, weighted=False):
    model.train()
    if freeze_bn:  # H15: keep BatchNorm running stats frozen (ImageNet stats)
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
    total_loss = correct = total = 0

    n_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        if weighted:
            images, labels, sw = batch
            sw = sw.to(DEVICE, non_blocking=True).float()
        else:
            images, labels = batch
            sw = None
        images, labels = images.to(DEVICE, non_blocking=True), \
                         labels.to(DEVICE, non_blocking=True)

        with torch.autocast(device_type="cuda",
                            enabled=DEVICE.type == "cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels, sample_weight=sw)

        # Scale loss for gradient accumulation (effective batch = micro x ACCUM).
        scaler.scale(loss / ACCUM_STEPS).backward()

        if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == n_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, return_probs=False):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE, non_blocking=True), \
                             labels.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type="cuda",
                                enabled=DEVICE.type == "cuda"):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro",
                        zero_division=0)
    if return_probs:
        return (total_loss / total, correct / total, macro_f1,
                all_preds, all_labels, np.array(all_probs))
    return total_loss / total, correct / total, macro_f1, all_preds, all_labels

# =============================================================================
# METRICS — comprehensive
# =============================================================================

def compute_all_metrics(labels, preds, probs, run_name):
    """
    Computes and saves every metric we need for the paper.
    Returns dict of scalar metrics.
    """
    results_dir = RESULTS_DIR / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    labels_bin = label_binarize(labels, classes=list(range(len(CLASSES))))

    # ── Classification report ─────────────────────────────────────────────
    report = classification_report(
        labels, preds,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(
        results_dir / "classification_report.csv"
    )

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(labels, preds)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        results_dir / "confusion_matrix.csv"
    )

    # ── ROC AUC per class ─────────────────────────────────────────────────
    roc_aucs = {}
    for i, cls in enumerate(CLASSES):
        try:
            roc_aucs[cls] = roc_auc_score(labels_bin[:, i], probs[:, i])
        except Exception:
            roc_aucs[cls] = 0.0
    roc_aucs["macro"] = np.mean(list(roc_aucs.values()))

    # ── PR AUC per class ──────────────────────────────────────────────────
    pr_aucs = {}
    for i, cls in enumerate(CLASSES):
        try:
            pr_aucs[cls] = average_precision_score(
                labels_bin[:, i], probs[:, i]
            )
        except Exception:
            pr_aucs[cls] = 0.0
    pr_aucs["macro"] = np.mean(list(pr_aucs.values()))

    # ── Kappa + MCC ───────────────────────────────────────────────────────
    kappa = cohen_kappa_score(labels, preds)
    mcc   = matthews_corrcoef(labels, preds)

    # ── Scalar summary ────────────────────────────────────────────────────
    scalars = {
        "run_name":        run_name,
        "test_acc":        round(np.mean(np.array(preds) == np.array(labels)), 4),
        "macro_f1":        round(report["macro avg"]["f1-score"], 4),
        "weighted_f1":     round(report["weighted avg"]["f1-score"], 4),
        "macro_roc_auc":   round(roc_aucs["macro"], 4),
        "macro_pr_auc":    round(pr_aucs["macro"], 4),
        "cohen_kappa":     round(kappa, 4),
        "mcc":             round(mcc, 4),
        **{f"f1_{cls}":       round(report[cls]["f1-score"], 4)
           for cls in CLASSES},
        **{f"precision_{cls}": round(report[cls]["precision"], 4)
           for cls in CLASSES},
        **{f"recall_{cls}":    round(report[cls]["recall"], 4)
           for cls in CLASSES},
        **{f"roc_auc_{cls}":   round(roc_aucs[cls], 4)
           for cls in CLASSES},
        **{f"pr_auc_{cls}":    round(pr_aucs[cls], 4)
           for cls in CLASSES},
    }
    pd.DataFrame([scalars]).to_csv(
        results_dir / "metrics_summary.csv", index=False
    )

    # ── Per-class ROC curve data ──────────────────────────────────────────
    roc_data = {}
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_data[cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    pd.DataFrame({
        f"{cls}_fpr": roc_data[cls]["fpr"]
        for cls in CLASSES
        if len(roc_data[cls]["fpr"]) == len(roc_data[CLASSES[0]]["fpr"])
    }).to_csv(results_dir / "roc_curves.csv", index=False)

    return scalars, cm, report, roc_aucs, pr_aucs, roc_data

# =============================================================================
# PLOTS — paper-quality
# =============================================================================

def save_confusion_matrix_plot(cm, run_name):
    results_dir = RESULTS_DIR / run_name
    fig, ax = plt.subplots(figsize=(7, 6))

    # Normalise to percentages
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax,
        linewidths=0.5,
        vmin=0, vmax=100,
        cbar_kws={"label": "% of True Class"}
    )

    # Add % symbol to each cell
    for text in ax.texts:
        text.set_text(text.get_text() + "%")

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(
        f"Confusion Matrix — {run_name.split('_')[1]}",
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        results_dir / "confusion_matrix.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()


def save_roc_curves(roc_data, roc_aucs, run_name):
    results_dir = RESULTS_DIR / run_name
    colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800"]
    fig, ax = plt.subplots(figsize=(7, 6))

    for cls, color in zip(CLASSES, colors):
        fpr = roc_data[cls]["fpr"]
        tpr = roc_data[cls]["tpr"]
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls} (AUC = {roc_aucs[cls]:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {run_name}\n"
                 f"Macro AUC = {roc_aucs['macro']:.3f}",
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "roc_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close()


def save_pr_curves(labels, probs, pr_aucs, run_name):
    results_dir = RESULTS_DIR / run_name
    labels_bin  = label_binarize(labels, classes=list(range(len(CLASSES))))
    colors      = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (cls, color) in enumerate(zip(CLASSES, colors)):
        prec, rec, _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{cls} (AP = {pr_aucs[cls]:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curves — {run_name}\n"
                 f"Macro AP = {pr_aucs['macro']:.3f}",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "pr_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close()


def save_per_class_bars(report, run_name):
    results_dir = RESULTS_DIR / run_name
    metrics     = ["precision", "recall", "f1-score"]
    colors      = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(CLASSES))

    for ax, metric, color in zip(axes, metrics, colors):
        vals = [report[cls][metric] for cls in CLASSES]
        bars = ax.bar(x, vals, color=color, alpha=0.8, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES, fontsize=11)
        ax.set_ylim(0.5, 1.05)
        ax.set_title(metric.replace("-", " ").title(), fontweight="bold")
        ax.set_ylabel("Score")
        ax.axhline(y=np.mean(vals), color="red", linestyle="--",
                   linewidth=1.2, alpha=0.7,
                   label=f"Mean: {np.mean(vals):.3f}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontsize=9)

    plt.suptitle(f"Per-Class Metrics — {run_name}",
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(results_dir / "per_class_metrics.png", dpi=150,
                bbox_inches="tight")
    plt.close()


def save_training_curves(history_df, run_name):
    results_dir = RESULTS_DIR / run_name
    fig, axes   = plt.subplots(1, 3, figsize=(16, 5))

    # Loss
    axes[0].plot(history_df["epoch"], history_df["train_loss"],
                 label="Train", color="#2196F3", lw=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"],
                 label="Val", color="#E91E63", lw=2)
    axes[0].set_title("Loss Curve", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Focal Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history_df["epoch"], history_df["train_acc"],
                 label="Train", color="#2196F3", lw=2)
    axes[1].plot(history_df["epoch"], history_df["val_acc"],
                 label="Val", color="#E91E63", lw=2)
    axes[1].set_title("Accuracy Curve", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Val F1
    best_ep = history_df.loc[history_df["val_f1"].idxmax(), "epoch"]
    best_f1 = history_df["val_f1"].max()
    axes[2].plot(history_df["epoch"], history_df["val_f1"],
                 color="#4CAF50", lw=2, label="Val Macro F1")
    axes[2].axvline(x=best_ep, color="red", linestyle="--",
                    lw=1.5, label=f"Best epoch {int(best_ep)}")
    axes[2].set_title(f"Val F1 — Best: {best_f1:.4f} @ ep{int(best_ep)}",
                      fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle(f"Training Curves — {run_name}",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close()


def save_class_probability_dist(labels, probs, run_name):
    """Confidence distribution per class — shows model calibration."""
    results_dir = RESULTS_DIR / run_name
    fig, axes   = plt.subplots(2, 2, figsize=(10, 8))
    axes        = axes.flatten()
    colors      = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800"]

    labels_arr = np.array(labels)
    for i, (cls, color) in enumerate(zip(CLASSES, colors)):
        cls_idx   = i
        cls_mask  = labels_arr == cls_idx
        correct   = probs[cls_mask, cls_idx]
        incorrect = probs[~cls_mask, cls_idx]

        axes[i].hist(correct, bins=30, alpha=0.7, color=color,
                     label=f"True {cls}", density=True)
        axes[i].hist(incorrect, bins=30, alpha=0.4, color="gray",
                     label=f"Other classes", density=True)
        axes[i].set_title(f"{cls} — Confidence Distribution",
                          fontweight="bold")
        axes[i].set_xlabel("Predicted Probability")
        axes[i].set_ylabel("Density")
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)

    plt.suptitle(f"Class Probability Distributions — {run_name}",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "probability_distributions.png", dpi=150,
                bbox_inches="tight")
    plt.close()

# =============================================================================
# MAIN
# =============================================================================

def main():
    global SEED, MICRO_BATCH, ACCUM_STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=str, required=True,
        choices=["A", "B", "C"],
        help="A=PTB-XL only  B=PTB-XL+Imagen  C=PTB-XL+Imagen+NeuroKit2"
    )
    parser.add_argument(
        "--fold", type=int, choices=[0, 1, 2], default=None,
        help="CV fold number (0-2). If not specified, uses legacy single split."
    )
    # --- Diagnostic-experiment flags (H1/H2/minority-aug) --------------------
    parser.add_argument("--real-weights", action="store_true",
                        help="H1: focal class weights from REAL train counts only")
    parser.add_argument("--synth-cap", type=int, default=None,
                        help="H2: max synthetic images per class per source")
    parser.add_argument("--aug-classes", type=str, default=None,
                        help="Minority-only aug: comma list of classes that get "
                             "synthetic, e.g. AFIB,TACHY (others stay real-only)")
    parser.add_argument("--finetune-real-epochs", type=int, default=0,
                        help="H13 two-stage: train the LAST N epochs on real-only")
    parser.add_argument("--synth-loss-weight", type=float, default=None,
                        help="H16: per-sample loss weight for synthetic images "
                             "(real=1.0), e.g. 0.3")
    parser.add_argument("--freeze-bn", action="store_true",
                        help="H15: freeze BatchNorm running stats during training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed. Different seeds give different CV "
                             "partitions -> repeated CV for tighter significance.")
    parser.add_argument("--arch", type=str, default="b0",
                        choices=["b0", "b1", "convnext_tiny"],
                        help="Backbone: EfficientNet-B0/B1 or ConvNeXt-Tiny")
    parser.add_argument("--micro-batch", type=int, default=MICRO_BATCH,
                        help="Per-step batch; accumulation keeps effective batch ~32")
    parser.add_argument("--no-gradcam", action="store_true",
                        help="Skip the auto Grad-CAM subprocess")
    args = parser.parse_args()

    # Repeated-CV: seed drives the fold partition, model init, and shuffling.
    SEED = args.seed
    MICRO_BATCH = args.micro_batch
    ACCUM_STEPS = max(1, round(32 / MICRO_BATCH))  # keep effective batch ~32

    aug_classes = ([c.strip().upper() for c in args.aug_classes.split(",")]
                   if args.aug_classes else None)

    exp_cfg  = EXPERIMENTS[args.experiment]
    fold_suffix = f"_fold{args.fold}" if args.fold is not None else ""
    # Variant tag keeps diagnostic runs in their own result dirs (expv_ prefix),
    # so they never collide with or pollute the canonical exp_A/B/C results.
    tag = ""
    if args.arch != "b0":      tag += f"_{args.arch}"
    if args.seed != 42:        tag += f"_s{args.seed}"
    if args.real_weights:      tag += "_rw"
    if args.synth_cap is not None: tag += f"_sc{args.synth_cap}"
    if aug_classes:            tag += "_" + "".join(c[0] for c in aug_classes)
    if args.finetune_real_epochs: tag += f"_ft{args.finetune_real_epochs}"
    if args.synth_loss_weight is not None: tag += f"_slw{int(args.synth_loss_weight*100)}"
    if args.freeze_bn:         tag += "_fbn"
    prefix = "expv_" if tag else "exp_"
    run_name = f"{prefix}{exp_cfg['name']}_img{IMG_SIZE}_bs{BATCH_SIZE}_e{EPOCHS}{tag}{fold_suffix}"

    print("\n" + "=" * 60)
    print(f"Experiment {args.experiment}: {exp_cfg['desc']}")
    if args.fold is not None:
        print(f"Cross-validation mode: FOLD {args.fold}/2 (3-fold CV)")
    else:
        print(f"Legacy mode: Single split (no CV)")
    print(f"Run name:  {run_name}")
    print(f"Device:    {DEVICE}")
    print(f"Img size:  {IMG_SIZE}px | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
    print("=" * 60)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # Keep cudnn.benchmark OFF: on the 6GB card its algorithm search inflates
    # peak memory enough to OOM EfficientNet-B0 at 512px/batch-32. The 512px
    # cache already unstarves the GPU, so we don't need autotuning for speed.
    torch.backends.cudnn.benchmark = False

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / run_name).mkdir(parents=True, exist_ok=True)

    # Dataset
    df = collect_images(exp_cfg["sources"], synth_cap=args.synth_cap,
                        aug_classes=aug_classes)
    print(f"\nTotal images: {len(df)}")
    print("Per source:")
    for src in df["source"].unique():
        sub = df[df["source"] == src]
        print(f"  {src}: {len(sub)}")
    print("Per class:")
    for cls in CLASSES:
        n = len(df[df["label"] == cls])
        print(f"  {cls}: {n}")

    # Build the one-time 512px cache so per-epoch decode is cheap.
    build_train_cache(df, workers=max(2, NUM_WORKERS))

    train_df, val_df, test_df = make_splits(df, run_name, fold_num=args.fold)

    # H16: per-sample loss weight (synthetic down-weighted, real=1.0).
    weighted = args.synth_loss_weight is not None
    if weighted:
        train_df = train_df.copy()
        train_df["sample_weight"] = np.where(
            train_df["source"] == "ptbxl", 1.0, args.synth_loss_weight)

    train_tf, val_tf = get_transforms()
    train_ds = ECGDataset(train_df, train_tf, use_weights=weighted)
    val_ds   = ECGDataset(val_df,   val_tf)
    test_ds  = ECGDataset(test_df,  val_tf)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, batch_size=MICRO_BATCH, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        worker_init_fn=seed_worker, generator=loader_gen
    )

    # H13 two-stage: a real-only training loader for the fine-tune phase.
    real_train_loader = None
    if args.finetune_real_epochs:
        real_train_df = train_df[train_df["source"] == "ptbxl"].reset_index(drop=True)
        real_train_ds = ECGDataset(real_train_df, train_tf, use_weights=weighted)
        real_train_loader = DataLoader(
            real_train_ds, batch_size=MICRO_BATCH, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False,
            worker_init_fn=seed_worker, generator=loader_gen
        )
    val_loader = DataLoader(
        val_ds, batch_size=MICRO_BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0
    )
    test_loader = DataLoader(
        test_ds, batch_size=MICRO_BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0
    )

    # Model + loss
    model        = build_model(args.arch)
    class_weights = compute_class_weights(train_df, real_only=args.real_weights)
    print(f"Class weights ({'real-only' if args.real_weights else 'train'} counts): "
          + ", ".join(f"{c}={w:.3f}" for c, w in zip(CLASSES, class_weights.tolist())))
    criterion    = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer    = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # Training
    best_val_f1 = 0
    best_state = None
    best_epoch = 0
    history = []
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    print(f"\nTraining {EPOCHS} epochs...\n")

    ft_start = EPOCHS - args.finetune_real_epochs + 1  # first real-only epoch
    for epoch in range(1, EPOCHS + 1):
        # H13: last N epochs fine-tune on real-only data.
        if real_train_loader is not None and epoch >= ft_start:
            loader = real_train_loader
            if epoch == ft_start:
                print(f"  [two-stage] epoch {epoch}: switching to REAL-only fine-tune")
        else:
            loader = train_loader
        train_loss, train_acc = train_epoch(
            model, loader, optimizer, criterion, scaler,
            freeze_bn=args.freeze_bn, weighted=weighted
        )
        val_loss, val_acc, val_f1, _, _ = eval_epoch(
            model, val_loader, criterion
        )
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   5),
            "val_acc":    round(val_acc,    4),
            "val_f1":     round(val_f1,     4),
            "lr":         round(lr_now,     8),
        })

        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone()
                           for k, v in model.state_dict().items()}
            best_epoch  = epoch
            marker      = " [BEST]"

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"TLoss:{train_loss:.4f} TAcc:{train_acc:.3f} | "
            f"VLoss:{val_loss:.4f} VAcc:{val_acc:.3f} VF1:{val_f1:.4f}"
            f"{marker}"
        )

    print(f"\nBest: epoch {best_epoch}, val F1 = {best_val_f1:.4f}")

    # Save training history
    hist_df   = pd.DataFrame(history)
    hist_path = RESULTS_DIR / run_name / "training_history.csv"
    hist_df.to_csv(hist_path, index=False)
    save_training_curves(hist_df, run_name)

    # Save model
    model_path = MODELS_DIR / f"{run_name}.pth"
    torch.save({
        "state_dict": best_state,
        "run_name":   run_name,
        "classes":    CLASSES,
        "img_size":   IMG_SIZE,
        "epoch":      best_epoch,
        "val_f1":     best_val_f1,
    }, model_path)
    print(f"Model saved: {model_path.name}")

    # Test evaluation
    print("\nEvaluating test set...")
    model.load_state_dict(best_state)
    model.to(DEVICE)

    _, test_acc, test_f1, preds, labels, probs = eval_epoch(
        model, test_loader, criterion, return_probs=True
    )

    # All metrics + plots
    scalars, cm, report, roc_aucs, pr_aucs, roc_data = compute_all_metrics(
        labels, preds, probs, run_name
    )

    # NEW: Calibration metrics
    ece = calibration_metrics.compute_ece(labels, probs, n_bins=10)
    calibration_metrics.plot_reliability_diagram(labels, probs, run_name, results_dir=RESULTS_DIR / run_name)
    calibration_metrics.plot_confidence_histogram(labels, probs, run_name, class_names=CLASSES, results_dir=RESULTS_DIR / run_name)
    confidence_stats = calibration_metrics.compute_confidence_stats(labels, probs, class_names=CLASSES)

    # Save predictions with confidence scores
    test_filepaths = test_df['filepath'].values
    calibration_metrics.save_confidence_predictions(labels, probs, test_filepaths, CLASSES, run_name, results_dir=RESULTS_DIR / run_name)

    # Add ECE to scalars and re-save summary (compute_all_metrics wrote it
    # before ECE existed, so overwrite with the complete row).
    scalars['ece'] = round(ece, 4)
    pd.DataFrame([scalars]).to_csv(
        RESULTS_DIR / run_name / "metrics_summary.csv", index=False
    )

    save_confusion_matrix_plot(cm, run_name)
    save_roc_curves(roc_data, roc_aucs, run_name)
    save_pr_curves(labels, probs, pr_aucs, run_name)
    save_per_class_bars(report, run_name)
    save_class_probability_dist(labels, probs, run_name)

    # Print results
    print("\n" + "=" * 60)
    print(f"EXPERIMENT {args.experiment} RESULTS")
    print("=" * 60)
    print(f"  Test Accuracy  : {scalars['test_acc']:.4f}")
    print(f"  Macro F1       : {scalars['macro_f1']:.4f}")
    print(f"  Weighted F1    : {scalars['weighted_f1']:.4f}")
    print(f"  Macro ROC-AUC  : {scalars['macro_roc_auc']:.4f}")
    print(f"  Macro PR-AUC   : {scalars['macro_pr_auc']:.4f}")
    print(f"  Cohen Kappa    : {scalars['cohen_kappa']:.4f}")
    print(f"  MCC            : {scalars['mcc']:.4f}")
    print(f"  ECE            : {scalars['ece']:.4f}")
    print("\nPer-class F1:")
    for cls in CLASSES:
        print(f"  {cls:6s}: F1={scalars[f'f1_{cls}']:.4f}  "
              f"ROC-AUC={scalars[f'roc_auc_{cls}']:.4f}  "
              f"PR-AUC={scalars[f'pr_auc_{cls}']:.4f}")
    print(f"\nOutputs: {RESULTS_DIR / run_name}")
    print("=" * 60)

    # Free GPU memory before the Grad-CAM subprocess so it doesn't OOM on the
    # 6GB card (the trained model + optimizer state are still resident here).
    model.to("cpu")
    del optimizer, scheduler, scaler
    torch.cuda.empty_cache()

    # Auto-launch GRAD-CAM (skippable for robustness sweeps / non-b0 backbones)
    if args.no_gradcam:
        print("\n[skip] Grad-CAM (--no-gradcam)")
        return
    print("\nLaunching GRAD-CAM...")
    import subprocess
    subprocess.run([
        sys.executable,
        str(BASE_DIR / "src" / "explainability" / "gradcam.py"),
        "--model", str(model_path),
        "--split", str(BASE_DIR / "data" / "splits" / run_name / "test.csv"),
    ])


if __name__ == "__main__":
    main()