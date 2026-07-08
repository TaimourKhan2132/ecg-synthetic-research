# =============================================================================
# imagen_generate.py
# Generates ECG images using Gemini 3 Pro Image generation via Vertex AI.
# Reads prompts from prompts.csv, saves images with full metadata logging.
# Supports resume — skips already generated images.
# =============================================================================

import os
import time
import warnings
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROMPTS_CSV   = os.path.join(BASE_DIR, "src", "generation", "prompts", "prompts.csv")
OUTPUT_DIR    = os.path.join(BASE_DIR, "data", "rendered", "imagen")
METADATA_FILE = os.path.join(BASE_DIR, "metadata", "imagen_generated.csv")

# =============================================================================
# GENERATION CONFIG
# =============================================================================

IMAGES_PER_PROMPT = 20   # 8 prompts x 20 = 160 per class = 640 total
MAX_RETRIES       = 5
RETRY_DELAY       = 15   # seconds between non-quota retries
QUOTA_BASE_DELAY  = 60   # seconds, multiplied by attempt on quota hits
REQUEST_INTERVAL  = 4    # seconds between successful requests (quota pacing)

# Correct model name for Gemini image generation on Vertex AI
MODEL_NAME = "gemini-3-pro-image"

# =============================================================================
# CLIENT INIT
# =============================================================================

def init_client():
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT missing from .env")
    print(f"  Project: {project_id}")
    print(f"  Model:   {MODEL_NAME}")
    return genai.Client(vertexai=True, project=project_id, location="global")

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

def setup_directories(classes):
    for c in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, c), exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)

# =============================================================================
# IMAGE GENERATION WITH RETRY
# =============================================================================

def generate_image(client, prompt_text):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    image_config=types.ImageConfig(
                        aspect_ratio="4:3",
                        image_size="1K",
                    ),
                )
            )

            for part in response.candidates[0].content.parts:
                if part.thought:
                    continue
                if part.inline_data is not None:
                    image_bytes = part.inline_data.data
                    image = Image.open(io.BytesIO(image_bytes))
                    return image

            print(f"\n  [WARN] No image in response. Attempt {attempt+1}/{MAX_RETRIES}")
            time.sleep(RETRY_DELAY)

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower() or "resource_exhausted" in error_msg.lower():
                sleep_time = QUOTA_BASE_DELAY * (attempt + 1)
                print(f"\n  [QUOTA] Backing off {sleep_time}s... (attempt {attempt+1})")
                time.sleep(sleep_time)
            elif "500" in error_msg or "503" in error_msg:
                print(f"\n  [SERVER] Temporary server error. Retry in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"\n  [ERROR] {error_msg[:120]}")
                if attempt == MAX_RETRIES - 1:
                    return None
                time.sleep(RETRY_DELAY)

    return None

# =============================================================================
# IMAGE VALIDATION
# =============================================================================

def validate_image(image: Image.Image) -> bool:
    """
    Basic sanity check on generated image.
    Rejects pure white/black outputs that indicate generation failure.
    """
    if image is None:
        return False

    import numpy as np
    arr = np.array(image.convert("RGB"))

    # Reject if image is essentially blank (all one color)
    std = arr.std()
    if std < 5.0:
        return False

    # Reject if image is too small
    if image.width < 100 or image.height < 100:
        return False

    return True

# =============================================================================
# METADATA MANAGEMENT
# =============================================================================

def load_metadata():
    if os.path.exists(METADATA_FILE):
        df = pd.read_csv(METADATA_FILE)
        # Build set of already completed signatures
        done = set(df["prompt_id"].astype(str) + "_" + df["iteration"].astype(str))
        return df, done
    else:
        df = pd.DataFrame(columns=[
            "class", "prompt_id", "prompt_text",
            "iteration", "filename", "filepath",
            "model", "status", "timestamp"
        ])
        return df, set()

def save_metadata(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df.to_csv(METADATA_FILE, index=False)

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 60)
    print("ECG Image Generator — Gemini 3 Pro Image via Vertex AI")
    print("=" * 60)

    # Init
    print("\n[1/4] Initializing client...")
    client = init_client()

    # Load prompts
    print("\n[2/4] Loading prompts...")
    if not os.path.exists(PROMPTS_CSV):
        raise FileNotFoundError(f"Prompts CSV not found: {PROMPTS_CSV}")

    prompts_df = pd.read_csv(PROMPTS_CSV)
    classes    = prompts_df["class"].unique().tolist()
    print(f"  Prompts loaded: {len(prompts_df)}")
    print(f"  Classes: {classes}")
    print(f"  Images per prompt: {IMAGES_PER_PROMPT}")
    print(f"  Total target: {len(prompts_df) * IMAGES_PER_PROMPT} images")

    # Setup dirs
    setup_directories(classes)

    # Load existing metadata for resume support
    print("\n[3/4] Checking existing progress...")
    metadata_df, done_signatures = load_metadata()
    print(f"  Already generated: {len(done_signatures)} images")

    # Calculate remaining work
    total_tasks = len(prompts_df) * IMAGES_PER_PROMPT
    remaining   = total_tasks - len(done_signatures)
    print(f"  Remaining: {remaining} images")

    if remaining == 0:
        print("\nAll images already generated. Nothing to do.")
        return

    # Generation loop
    print("\n[4/4] Generating images...")
    success_count = 0
    fail_count    = 0
    new_rows      = []

    with tqdm(total=total_tasks, initial=len(done_signatures),
              desc="Generating", unit="img", ncols=80) as pbar:

        for _, row in prompts_df.iterrows():
            cls     = row["class"]
            p_id    = row["prompt_id"]
            p_text  = row["prompt_text"]

            for i in range(IMAGES_PER_PROMPT):
                signature = f"{p_id}_{i}"

                # Resume: skip already done
                if signature in done_signatures:
                    continue

                # Generate
                image = generate_image(client, p_text)

                timestamp = pd.Timestamp.now().isoformat()
                filename  = f"nano_{cls}_{p_id}_{i:03d}.png"
                filepath  = os.path.join(OUTPUT_DIR, cls, filename)

                if image and validate_image(image):
                    # Convert to RGB and save as PNG
                    image.convert("RGB").save(filepath, format="PNG")

                    new_rows.append({
                        "class":       cls,
                        "prompt_id":   p_id,
                        "prompt_text": p_text,
                        "iteration":   i,
                        "filename":    filename,
                        "filepath":    filepath,
                        "model":       MODEL_NAME,
                        "status":      "success",
                        "timestamp":   timestamp,
                    })
                    done_signatures.add(signature)
                    success_count += 1

                else:
                    new_rows.append({
                        "class":       cls,
                        "prompt_id":   p_id,
                        "prompt_text": p_text,
                        "iteration":   i,
                        "filename":    filename,
                        "filepath":    "",
                        "model":       MODEL_NAME,
                        "status":      "failed",
                        "timestamp":   timestamp,
                    })
                    fail_count += 1

                # Save metadata every 10 images
                if len(new_rows) % 10 == 0:
                    metadata_df = pd.concat(
                        [metadata_df, pd.DataFrame(new_rows[-10:])],
                        ignore_index=True
                    )
                    save_metadata(metadata_df)

                pbar.update(1)

                # Pace requests
                time.sleep(REQUEST_INTERVAL)

    # Final metadata flush
    if new_rows:
        metadata_df = pd.concat(
            [metadata_df, pd.DataFrame(new_rows)],
            ignore_index=True
        ).drop_duplicates(subset=["prompt_id", "iteration"])
        save_metadata(metadata_df)

    # Summary
    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Success:  {success_count}")
    print(f"  Failed:   {fail_count}")
    print(f"  Metadata: {METADATA_FILE}")
    print("\nFinal class distribution:")
    success_df = metadata_df[metadata_df["status"] == "success"]
    print(success_df["class"].value_counts().to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()