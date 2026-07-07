# =============================================================================
# gradcam_paper.py — paper-quality, curated Grad-CAM per class.
# For each class: pick the highest-confidence CORRECTLY-classified test samples,
# overlay the class-discriminative heatmap, and add a ZOOMED crop of the
# peak-activation region ("which area the class targets most"). High DPI.
# Output: outputs/figures_paper/gradcam/gradcam_<CLASS>.png (+ overview)
# =============================================================================
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src")); sys.path.insert(0, str(BASE / "src" / "training"))
import train as T

CLASSES = T.CLASSES
OUT = BASE / "outputs" / "figures_paper" / "gradcam"; OUT.mkdir(parents=True, exist_ok=True)
DEVICE = T.DEVICE
N = 3            # correct samples per class
ZOOM = 150       # crop half-window (px) around peak activation


def tf():
    return transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def to_rgb(t):
    m = np.array([0.485, 0.456, 0.406]); s = np.array([0.229, 0.224, 0.225])
    return np.clip(s * t.permute(1, 2, 0).cpu().numpy() + m, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--arch", default="b0")
    a = ap.parse_args()

    ck = torch.load(a.model, map_location=DEVICE)
    model = T.build_model(a.arch); model.load_state_dict(ck["state_dict"]); model.eval().to(DEVICE)
    transform = tf()
    df = pd.read_csv(a.split); df = df[df["source"] == "ptbxl"].reset_index(drop=True)
    c2i = {c: i for i, c in enumerate(CLASSES)}
    cam = GradCAM(model=model, target_layers=[model.features[-1] if a.arch != "convnext_tiny"
                                              else model.features[-1]])

    heroes = []
    for cls in CLASSES:
        idx = c2i[cls]
        sub = df[df["label"] == cls].reset_index(drop=True)
        scored = []
        for _, row in sub.iterrows():
            img = Image.open(row["filepath"]).convert("RGB"); x = transform(img)
            with torch.no_grad():
                out = model(x.unsqueeze(0).to(DEVICE)); p = torch.softmax(out, 1)[0]
            if out.argmax().item() == idx:
                scored.append((p[idx].item(), row["filepath"]))
        scored.sort(reverse=True)
        picks = scored[:N]

        fig, axes = plt.subplots(3, len(picks), figsize=(len(picks) * 6, 15))
        if len(picks) == 1:
            axes = axes.reshape(3, 1)
        fig.suptitle(f"Grad-CAM — {cls}", fontsize=30, fontweight="bold", y=1.005)
        for col, (conf, fp) in enumerate(picks):
            img = Image.open(fp).convert("RGB"); x = transform(img); rgb = to_rgb(x)
            g = cam(input_tensor=x.unsqueeze(0), targets=[ClassifierOutputTarget(idx)])[0]
            overlay = show_cam_on_image(rgb, g, use_rgb=True)
            # peak region
            yy, xx = np.unravel_index(np.argmax(g), g.shape)
            y0, y1 = max(0, yy - ZOOM), min(512, yy + ZOOM); x0, x1 = max(0, xx - ZOOM), min(512, xx + ZOOM)
            axes[0, col].imshow(rgb); axes[0, col].set_title(f"ECG (conf {conf:.2f})", fontsize=18, fontweight="bold"); axes[0, col].axis("off")
            axes[1, col].imshow(overlay); axes[1, col].set_title("Grad-CAM", fontsize=18, fontweight="bold"); axes[1, col].axis("off")
            axes[2, col].imshow(overlay[y0:y1, x0:x1]); axes[2, col].set_title("Zoom: peak activation", fontsize=18, fontweight="bold"); axes[2, col].axis("off")
            if col == 0:
                heroes.append((cls, overlay))
        plt.tight_layout()
        fig.savefig(OUT / f"gradcam_{cls}.png", dpi=400, bbox_inches="tight")
        fig.savefig(OUT / f"gradcam_{cls}.pdf", bbox_inches="tight"); plt.close(fig)
        print(f"  saved gradcam_{cls} ({len(picks)} correct samples)")

    # overview: one hero per class
    fig, axes = plt.subplots(1, len(heroes), figsize=(len(heroes) * 6, 6.5))
    for ax, (cls, ov) in zip(axes, heroes):
        ax.imshow(ov); ax.set_title(cls, fontsize=22, fontweight="bold"); ax.axis("off")
    fig.suptitle("Grad-CAM overview (correct, high-confidence)", fontsize=22, fontweight="bold")
    plt.tight_layout(); fig.savefig(OUT / "gradcam_overview.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / "gradcam_overview.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"  saved gradcam_overview\nDone -> {OUT}")


if __name__ == "__main__":
    main()
