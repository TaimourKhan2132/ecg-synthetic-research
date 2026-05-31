# Masks top 30 pixels of every NeuroKit2 image with the background color
# Eliminates condition name / HR / timestamp text strip
# Runs in-place on existing images

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

NK_DIR     = Path(r"C:\Users\taimo\OneDrive\Documents 1\work\ecg-synthetic-research\data\rendered\neurokit2")
CLASSES    = ["NORM"]
MASK_ROWS  = 35        # pixels to blank from top
PAPER_BG   = (255, 248, 231)  # #FFF8E7 as RGB

def fix_image(path: Path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    arr[:MASK_ROWS, :] = PAPER_BG
    Image.fromarray(arr).save(path)

def main():
    for cls in CLASSES:
        cls_dir = NK_DIR / cls
        if not cls_dir.exists():
            print(f"  [SKIP] {cls} folder not found")
            continue
        images = list(cls_dir.rglob("*.png"))
        print(f"  {cls}: {len(images)} images")
        for img_path in tqdm(images, desc=cls):
            try:
                fix_image(img_path)
            except Exception as e:
                print(f"\n  [ERROR] {img_path.name}: {e}")

if __name__ == "__main__":
    main()