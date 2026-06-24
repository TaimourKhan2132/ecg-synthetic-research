# =============================================================================
# gradcam.py — GRAD-CAM visualization for trained ECG models
# Usage: python src/explainability/gradcam.py --model outputs/models/xxx.pth
#        --split data/splits/xxx/test.csv
# Or called automatically from train.py
# =============================================================================

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

BASE_DIR    = Path(r"C:\Users\taimo\OneDrive\Documents 1\work\ecg-synthetic-research")
GRADCAM_DIR = BASE_DIR / "outputs" / "gradcam"
CLASSES     = ["NORM", "MI", "AFIB", "TACHY"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = BASE_DIR / "outputs" / "results"

SAMPLES_PER_CLASS = 3   # how many test images per class to visualize

# =============================================================================
# MODEL LOADER
# =============================================================================

def load_model(model_path: Path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    img_size   = checkpoint.get("img_size", 512)

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, len(CLASSES))
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(DEVICE)

    return model, img_size, checkpoint.get("run_name", model_path.stem)

# =============================================================================
# TRANSFORM
# =============================================================================

def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

# =============================================================================
# GRAD-CAM GENERATION
# =============================================================================

def generate_gradcam(model, img_tensor, class_idx):
    """Returns GRAD-CAM heatmap for given class."""
    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(
        input_tensor=img_tensor.unsqueeze(0),
        targets=targets
    )
    return grayscale_cam[0]

def tensor_to_rgb(img_tensor, img_size):
    """Denormalizes tensor back to [0,1] RGB numpy array."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = img_tensor.permute(1, 2, 0).cpu().numpy()
    img  = std * img + mean
    img  = np.clip(img, 0, 1)
    return img

# =============================================================================
# VISUALIZATION — grid of originals + overlays per class
# =============================================================================

def make_class_grid(model, transform, test_df, class_name, out_dir, img_size):
    class_df = test_df[test_df["label"] == class_name].sample(
        n=min(SAMPLES_PER_CLASS, len(test_df[test_df["label"] == class_name])),
        random_state=42
    )
    class_idx = CLASSES.index(class_name)
    n = len(class_df)

    # Larger figure — more space per image, readable text
    fig, axes = plt.subplots(2, n, figsize=(n * 6, 10))
    fig.suptitle(
        f"GRAD-CAM — Class: {class_name}",
        fontsize=22, fontweight="bold", y=1.01
    )

    for col, (_, row) in enumerate(class_df.iterrows()):
        img_pil    = Image.open(row["filepath"]).convert("RGB")
        img_tensor = transform(img_pil)
        img_rgb    = tensor_to_rgb(img_tensor, img_size)

        with torch.no_grad():
            out  = model(img_tensor.unsqueeze(0).to(DEVICE))
            pred = CLASSES[out.argmax(1).item()]
            conf = torch.softmax(out, dim=1).max().item()

        cam_map = generate_gradcam(model, img_tensor.to(DEVICE), class_idx)
        overlay = show_cam_on_image(img_rgb, cam_map, use_rgb=True)

        # Original
        axes[0, col].imshow(img_rgb)
        axes[0, col].set_title(
            f"Ground Truth: {class_name}",
            fontsize=14, fontweight="bold", pad=8
        )
        axes[0, col].axis("off")

        # GRAD-CAM overlay
        correct = "✓" if pred == class_name else "✗"
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(
            f"Pred: {pred} {correct}  |  Conf: {conf:.2f}",
            fontsize=13, pad=8,
            color="green" if pred == class_name else "red"
        )
        axes[1, col].axis("off")

    plt.tight_layout()
    out_path = out_dir / f"gradcam_{class_name}.png"
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return out_path

# =============================================================================
# COMPARISON GRID — side by side across experiments
# =============================================================================

def make_comparison_grid(gradcam_dirs: list, out_path: Path):
    """
    Given multiple gradcam output dirs, creates a side-by-side comparison
    grid for all classes. Called from dashboard.
    """
    pass  # Implemented in dashboard via image loading

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GRAD-CAM for ECG models")
    parser.add_argument("--model", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--split", required=True, help="Path to test.csv split")
    parser.add_argument(
        "--samples", type=int, default=SAMPLES_PER_CLASS,
        help="Images per class"
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    split_path = Path(args.split)

    print("\n" + "=" * 60)
    print(f"GRAD-CAM — {model_path.stem}")
    print("=" * 60)

    model, img_size, run_name = load_model(model_path)
    transform = get_transform(img_size)
    test_df   = pd.read_csv(split_path)

    out_dir = RESULTS_DIR / run_name / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating GRAD-CAM for {len(test_df)} test images...")
    print(f"Output: {out_dir}\n")

    generated = []
    for cls in CLASSES:
        print(f"  Class: {cls}")
        p = make_class_grid(
            model, transform, test_df, cls, out_dir, img_size
        )
        generated.append(p)

    # Combined 4-class overview
    fig, axes = plt.subplots(1, 4, figsize=(28, 8))
    fig.suptitle(
        f"GRAD-CAM Overview — {run_name}",
        fontsize=22, fontweight="bold"
    )
    for ax, cls in zip(axes, CLASSES):
        img_path = out_dir / f"gradcam_{cls}.png"
        if img_path.exists():
            img = Image.open(img_path)
            ax.imshow(img)
        ax.set_title(cls, fontsize=18, fontweight="bold", pad=10)
        ax.axis("off")

    overview_path = out_dir / "gradcam_overview.png"
    plt.tight_layout()
    plt.savefig(overview_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"\nOverview saved: {overview_path}")

    print("\n" + "=" * 60)
    print(f"GRAD-CAM complete — {len(CLASSES)} classes visualized")
    print(f"Output folder: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()