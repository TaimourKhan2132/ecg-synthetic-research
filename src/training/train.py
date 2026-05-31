# =============================================================================
# train.py
# Fixed experiments A, B, C. 512px. Focal loss. Class weights only.
# Maximum metrics: F1, ROC-AUC, PR-AUC, Kappa, MCC, confusion matrix.
# Usage: python src/training/train.py --experiment A
# =============================================================================

import os
import sys
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
BATCH_SIZE  = 32
EPOCHS      = 25
LR          = 1e-4
NUM_WORKERS = 2

# =============================================================================
# DATASET
# =============================================================================

def collect_images(sources: dict) -> pd.DataFrame:
    records = []
    for src_name, src_dir in sources.items():
        src_dir = Path(src_dir)
        for cls in CLASSES:
            cls_dir = src_dir / cls
            if not cls_dir.exists():
                print(f"  [WARN] Missing: {cls_dir}")
                continue
            images = list(cls_dir.rglob("*.png"))

            # Cap NeuroKit2
            if src_name == "neurokit2" and len(images) > NK_CAP:
                rng = np.random.RandomState(SEED)
                images = list(rng.choice(images, NK_CAP, replace=False))

            for img_path in images:
                records.append({
                    "filepath": str(img_path),
                    "label":    cls,
                    "source":   src_name,
                })

    df = pd.DataFrame(records)
    return df


def make_splits(df: pd.DataFrame, exp_name: str):
    split_dir = SPLITS_DIR / exp_name
    split_dir.mkdir(parents=True, exist_ok=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED
    )

    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir   / "val.csv",   index=False)
    test_df.to_csv(split_dir  / "test.csv",  index=False)

    print(f"\nSplit — Train:{len(train_df)} Val:{len(val_df)} Test:{len(test_df)}")
    print("Train class distribution:")
    for cls in CLASSES:
        n    = len(train_df[train_df["label"] == cls])
        pct  = 100 * n / len(train_df)
        print(f"  {cls:6s}: {n:5d} ({pct:.1f}%)")

    return train_df, val_df, test_df


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform):
        self.df           = df.reset_index(drop=True)
        self.transform    = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        image = self.transform(image)
        label = self.class_to_idx[row["label"]]
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

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets,
                                  weight=self.alpha, reduction="none")
        pt      = torch.exp(-ce_loss)
        focal   = (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


def compute_class_weights(train_df):
    class_to_idx  = {c: i for i, c in enumerate(CLASSES)}
    labels        = [class_to_idx[l] for l in train_df["label"]]
    counts        = np.bincount(labels, minlength=len(CLASSES))
    weights       = 1.0 / (counts + 1e-6)
    weights       = weights / weights.sum() * len(CLASSES)
    return torch.FloatTensor(weights).to(DEVICE)

# =============================================================================
# MODEL
# =============================================================================

def build_model():
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, len(CLASSES))
    )
    return model.to(DEVICE)

# =============================================================================
# TRAIN / EVAL LOOPS
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE, non_blocking=True), \
                         labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda",
                            enabled=DEVICE.type == "cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

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
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
        ax=ax, linewidths=0.5,
        cbar_kws={"label": "Count"}
    )
    # Overlay normalized percentages
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j + 0.5, i + 0.75,
                    f"({cm_norm[i, j]:.1%})",
                    ha="center", va="center",
                    fontsize=7, color="gray")

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {run_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150,
                bbox_inches="tight")
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=str, required=True,
        choices=["A", "B", "C"],
        help="A=PTB-XL only  B=PTB-XL+Imagen  C=PTB-XL+Imagen+NeuroKit2"
    )
    args = parser.parse_args()

    exp_cfg  = EXPERIMENTS[args.experiment]
    run_name = f"exp_{exp_cfg['name']}_img{IMG_SIZE}_bs{BATCH_SIZE}_e{EPOCHS}"

    print("\n" + "=" * 60)
    print(f"Experiment {args.experiment}: {exp_cfg['desc']}")
    print(f"Run name:  {run_name}")
    print(f"Device:    {DEVICE}")
    print(f"Img size:  {IMG_SIZE}px | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
    print("=" * 60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / run_name).mkdir(parents=True, exist_ok=True)

    # Dataset
    df = collect_images(exp_cfg["sources"])
    print(f"\nTotal images: {len(df)}")
    print("Per source:")
    for src in df["source"].unique():
        sub = df[df["source"] == src]
        print(f"  {src}: {len(sub)}")
    print("Per class:")
    for cls in CLASSES:
        n = len(df[df["label"] == cls])
        print(f"  {cls}: {n}")

    train_df, val_df, test_df = make_splits(df, run_name)

    train_tf, val_tf = get_transforms()
    train_ds = ECGDataset(train_df, train_tf)
    val_ds   = ECGDataset(val_df,   val_tf)
    test_ds  = ECGDataset(test_df,  val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0
    )

    # Model + loss
    model        = build_model()
    class_weights = compute_class_weights(train_df)
    criterion    = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer    = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=DEVICE.type == "cuda"
    )

    # Training
    best_val_f1 = 0
    best_state  = None
    best_epoch  = 0
    history     = []

    print(f"\nTraining {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler
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
            marker      = " ★ saved"

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
    print("\nPer-class F1:")
    for cls in CLASSES:
        print(f"  {cls:6s}: F1={scalars[f'f1_{cls}']:.4f}  "
              f"ROC-AUC={scalars[f'roc_auc_{cls}']:.4f}  "
              f"PR-AUC={scalars[f'pr_auc_{cls}']:.4f}")
    print(f"\nOutputs: {RESULTS_DIR / run_name}")
    print("=" * 60)

    # Auto-launch GRAD-CAM
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