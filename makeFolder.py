import os
import json
import shutil

# ----- CONFIG -----
INPUT_JSON = "imagenet_val_subset_1000.json"  # from previous script
OUTPUT_DIR = "sampled_1000_images_from5000"  # target folder

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the JSON
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

# Copy images
for entry in data:
    img_path = entry["image"]
    dest_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    try:
        shutil.copy(img_path, dest_path)
    except Exception as e:
        print(f"Failed to copy {img_path}: {e}")

print(f"\n✅ Copied {len(data)} images to '{OUTPUT_DIR}'")
