# =============================================================================
# sanitize_imagen.py
# Uses EasyOCR to detect and inpaint text in Imagen-generated ECG images.
# Skips standard ECG lead labels — only removes disease names / artifacts.
# Reads from data/rendered/imagen/ → writes to data/rendered/imagen_clean/
# =============================================================================

import cv2
import easyocr
import numpy as np
import os
from pathlib import Path

# Lead name whitelist — these are anatomical markers, NOT class leakage
# OCR detections matching these are skipped
LEAD_WHITELIST = {
    "i", "ii", "iii", "iv",
    "avr", "avl", "avf",
    "v1", "v2", "v3", "v4", "v5", "v6",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
    "I", "II", "III",
}

# Also skip very short single-character detections (grid artifacts)
MIN_TEXT_LENGTH = 2

# Confidence threshold — only mask text OCR is confident about
MIN_CONFIDENCE = 0.4

PROJECT_ROOT  = Path(r"C:\Users\taimo\OneDrive\Documents 1\work\ecg-synthetic-research")
INPUT_DIR     = PROJECT_ROOT / "data" / "rendered" / "imagen"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "rendered" / "imagen_clean"
CLASSES       = ["AFIB", "MI", "NORM", "TACHY"]
PADDING       = 8

def is_lead_label(text: str) -> bool:
    """Returns True if detected text is a standard ECG lead name."""
    cleaned = text.strip().upper().replace(".", "").replace(" ", "")
    return cleaned in {l.upper() for l in LEAD_WHITELIST}

def sanitize_image(reader, img_path: Path, out_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [SKIP] Cannot load: {img_path.name}")
        return False

    results = reader.readtext(img, detail=1)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    detections = 0

    for (bbox, text, prob) in results:
        if prob < MIN_CONFIDENCE:
            continue
        if len(text.strip()) < MIN_TEXT_LENGTH:
            continue
        if is_lead_label(text):
            continue  # preserve lead names

        (tl, tr, br, bl) = bbox
        x_min = max(0, int(tl[0]) - PADDING)
        y_min = max(0, int(tl[1]) - PADDING)
        x_max = min(img.shape[1], int(br[0]) + PADDING)
        y_max = min(img.shape[0], int(br[1]) + PADDING)
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
        detections += 1

    if np.any(mask):
        clean_img = cv2.inpaint(img, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    else:
        clean_img = img

    cv2.imwrite(str(out_path), clean_img)
    return detections

def main():
    print("=" * 60)
    print("Imagen ECG Sanitizer — EasyOCR + Inpainting")
    print("=" * 60)
    print("\nInitializing EasyOCR (GPU)...")
    reader = easyocr.Reader(['en'], gpu=True)

    total_processed = 0
    total_detections = 0

    for cls in CLASSES:
        in_dir  = INPUT_DIR  / cls
        out_dir = OUTPUT_DIR / cls

        if not in_dir.exists():
            print(f"\n[SKIP] {cls} — input folder not found")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        images = sorted([
            f for f in in_dir.iterdir()
            if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}
        ])

        print(f"\n--- {cls}: {len(images)} images ---")

        for img_path in images:
            out_path = out_dir / img_path.name

            if out_path.exists():
                continue  # resume support

            n = sanitize_image(reader, img_path, out_path)
            if n is not False:
                total_processed += 1
                total_detections += n
                if n > 0:
                    print(f"  {img_path.name}: {n} text regions removed")

    print("\n" + "=" * 60)
    print(f"DONE — {total_processed} images processed")
    print(f"       {total_detections} total text regions inpainted")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()