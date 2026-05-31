# Fix 1: Crop 15px from each edge (removes watermark-style borders)
# Fix 2: Mask top-left and top-right corners 40x120px (removes corner text)
# Saves fixed images back in-place

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

IMAGEN_DIR = Path(r"C:\Users\taimo\OneDrive\Documents 1\work\ecg-synthetic-research\data\rendered\imagen")
CLASSES    = ["NORM", "MI", "AFIB", "TACHY"]
EDGE_CROP  = 15       # pixels to crop from all 4 edges
CORNER_H   = 50       # height of corner mask
CORNER_W   = 180      # width of corner mask

def fix_image(path: Path):
    img = Image.open(path).convert("RGB")
    w, h = img.size

    # Step 1: Crop edges
    img = img.crop((EDGE_CROP, EDGE_CROP, w - EDGE_CROP, h - EDGE_CROP))
    arr = np.array(img)

    # Step 2: Mask top-left corner (condition name often here)
    arr[:CORNER_H, :CORNER_W] = 255  # white

    # Step 3: Mask top-right corner
    arr[:CORNER_H, -CORNER_W:] = 255  # white

    # Step 4: Mask bottom-left and bottom-right corners too
    arr[-CORNER_H:, :CORNER_W] = 255
    arr[-CORNER_H:, -CORNER_W:] = 255

    Image.fromarray(arr).save(path)

def main():
    for cls in CLASSES:
        cls_dir = IMAGEN_DIR / cls
        if not cls_dir.exists():
            print(f"  [SKIP] {cls} folder not found")
            continue
        images = list(cls_dir.rglob("*.png"))
        print(f"  {cls}: {len(images)} images")
        for img_path in tqdm(images, desc=cls):
            try:
                fix_image(path=img_path)
            except Exception as e:
                print(f"\n  [ERROR] {img_path.name}: {e}")

if __name__ == "__main__":
    main()