# =============================================================================
# train_cross_domain.py
# Experiments D & E: Cross-domain training (synthetic train, real test)
# Reuses all training infrastructure from train.py — train.py NOT modified
# =============================================================================

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Import from train.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.train import (
    CLASSES, SEED, DEVICE, IMG_SIZE, BATCH_SIZE, EPOCHS, LR, NUM_WORKERS,
    BASE_DIR, PTBXL_DIR, IMAGEN_DIR, NEUROKIT_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR,
    ECGDataset, get_transforms, FocalLoss, build_model,
    train_epoch, eval_epoch, compute_all_metrics,
    save_confusion_matrix_plot, save_roc_curves, save_pr_curves, save_per_class_bars,
    save_training_curves, save_class_probability_dist
)
from utils import calibration_metrics

# =============================================================================
# EXPERIMENTS D & E
# =============================================================================

EXPERIMENTS_CROSS_DOMAIN = {
    "D": {
        "name": "D_neurokit2_train_ptbxl_test",
        "train_source": {"neurokit2": NEUROKIT_DIR},
        "test_source": {"ptbxl": PTBXL_DIR},
        "desc": "Train: NeuroKit2 only | Test: PTB-XL only",
    },
    "E": {
        "name": "E_imagen_train_ptbxl_test",
        "train_source": {"imagen": IMAGEN_DIR},
        "test_source": {"ptbxl": PTBXL_DIR},
        "desc": "Train: Imagen only | Test: PTB-XL only",
    },
}

NK_CAP = 1500


def collect_images_cross_domain(train_sources, test_sources):
    """Collect images separately for train and test sets."""
    train_records = []
    test_records = []

    # Training images
    for src_name, src_dir in train_sources.items():
        src_dir = Path(src_dir)
        for cls in CLASSES:
            cls_dir = src_dir / cls
            if not cls_dir.exists():
                print(f"  [WARN] Missing train: {cls_dir}")
                continue
            images = list(cls_dir.rglob("*.png"))

            if src_name == "neurokit2" and len(images) > NK_CAP:
                rng = np.random.RandomState(SEED)
                images = list(rng.choice(images, NK_CAP, replace=False))

            for img_path in images:
                train_records.append({
                    "filepath": str(img_path),
                    "label": cls,
                    "source": src_name,
                })

    # Test images
    for src_name, src_dir in test_sources.items():
        src_dir = Path(src_dir)
        for cls in CLASSES:
            cls_dir = src_dir / cls
            if not cls_dir.exists():
                print(f"  [WARN] Missing test: {cls_dir}")
                continue
            images = list(cls_dir.rglob("*.png"))
            for img_path in images:
                test_records.append({
                    "filepath": str(img_path),
                    "label": cls,
                    "source": src_name,
                })

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)
    return train_df, test_df


def make_splits_cross_domain(train_df, test_df, exp_name):
    """
    Train:
        synthetic only

    Validation:
        PTB-XL subset

    Test:
        PTB-XL subset

    Target test size:
        250 images per class
    """

    split_dir = SPLITS_DIR / exp_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Synthetic training data
    # --------------------------------------------------

    train_final = train_df.copy()

    # --------------------------------------------------
    # PTB-XL -> Validation + Test
    # --------------------------------------------------

    test_parts = []
    remaining_parts = []

    rng = np.random.RandomState(SEED)

    for cls in CLASSES:

        cls_df = test_df[test_df["label"] == cls]

        n_test = min(250, len(cls_df))

        sampled_test = cls_df.sample(
            n=n_test,
            random_state=SEED
        )

        remaining = cls_df.drop(sampled_test.index)

        test_parts.append(sampled_test)
        remaining_parts.append(remaining)

    test_final = pd.concat(test_parts).reset_index(drop=True)

    remaining_real = pd.concat(
        remaining_parts
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Validation from remaining PTB-XL
    # --------------------------------------------------

    val_final, _ = train_test_split(
        remaining_real,
        test_size=0.50,
        stratify=remaining_real["label"],
        random_state=SEED
    )

    train_final.to_csv(
        split_dir / "train.csv",
        index=False
    )

    val_final.to_csv(
        split_dir / "val.csv",
        index=False
    )

    test_final.to_csv(
        split_dir / "test.csv",
        index=False
    )

    print(
        f"\nSplit — "
        f"Train:{len(train_final)} "
        f"Val:{len(val_final)} "
        f"Test:{len(test_final)}"
    )

    return train_final, val_final, test_final


def compute_class_weights(train_df):
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    labels = [class_to_idx[l] for l in train_df["label"]]
    counts = np.bincount(labels, minlength=len(CLASSES))
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(CLASSES)
    return torch.FloatTensor(weights).to(DEVICE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=str, required=True,
        choices=["D", "E"],
        help="D=Train NeuroKit2 test PTB-XL  E=Train Imagen test PTB-XL"
    )
    args = parser.parse_args()

    exp_cfg = EXPERIMENTS_CROSS_DOMAIN[args.experiment]
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

    # Collect cross-domain data
    train_df, test_df = collect_images_cross_domain(
        exp_cfg["train_source"], exp_cfg["test_source"]
    )

    print(f"\nTrain images: {len(train_df)}")
    print("Train per source:")
    for src in train_df["source"].unique():
        sub = train_df[train_df["source"] == src]
        print(f"  {src}: {len(sub)}")
    print("Train per class:")
    for cls in CLASSES:
        n = len(train_df[train_df["label"] == cls])
        print(f"  {cls}: {n}")

    print(f"\nTest images: {len(test_df)}")
    print("Test per source:")
    for src in test_df["source"].unique():
        sub = test_df[test_df["source"] == src]
        print(f"  {src}: {len(sub)}")
    print("Test per class:")
    for cls in CLASSES:
        n = len(test_df[test_df["label"] == cls])
        print(f"  {cls}: {n}")

    # Create splits
    train_df_final, val_df, test_df_final = make_splits_cross_domain(
        train_df, test_df, run_name
    )

    # Dataset + loaders
    train_tf, val_tf = get_transforms()
    train_ds = ECGDataset(train_df_final, train_tf)
    val_ds = ECGDataset(val_df, val_tf)
    test_ds = ECGDataset(test_df_final, val_tf)

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
    model = build_model()
    class_weights = compute_class_weights(train_df_final)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(
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
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "lr": round(lr_now, 8),
        })

        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone()
                         for k, v in model.state_dict().items()}
            best_epoch = epoch
            marker = " [BEST]"

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"TLoss:{train_loss:.4f} TAcc:{train_acc:.3f} | "
            f"VLoss:{val_loss:.4f} VAcc:{val_acc:.3f} VF1:{val_f1:.4f}"
            f"{marker}"
        )

    print(f"\nBest: epoch {best_epoch}, val F1 = {best_val_f1:.4f}")

    # Save training history
    hist_df = pd.DataFrame(history)
    hist_path = RESULTS_DIR / run_name / "training_history.csv"
    hist_df.to_csv(hist_path, index=False)
    save_training_curves(hist_df, run_name)

    # Save model
    model_path = MODELS_DIR / f"{run_name}.pth"
    torch.save({
        "state_dict": best_state,
        "run_name": run_name,
        "classes": CLASSES,
        "img_size": IMG_SIZE,
        "epoch": best_epoch,
        "val_f1": best_val_f1,
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

    # Calibration metrics
    ece = calibration_metrics.compute_ece(labels, probs, n_bins=10)
    calibration_metrics.plot_reliability_diagram(
        labels, probs, run_name, results_dir=RESULTS_DIR / run_name
    )
    calibration_metrics.plot_confidence_histogram(
        labels, probs, run_name, class_names=CLASSES, results_dir=RESULTS_DIR / run_name
    )
    calibration_metrics.save_confidence_predictions(
        labels, probs, test_df_final['filepath'].values, CLASSES, run_name,
        results_dir=RESULTS_DIR / run_name
    )

    scalars['ece'] = round(ece, 4)

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


if __name__ == "__main__":
    main()
