"""
Task 1 (part a) — Split the Semantic Drone Dataset into train / valid.

Result (inside DATASET_ROOT/training_set):
    train/images/   80% of the images
    valid/images/   20% of the images
    gt/semantic/label_images/   masks (unchanged, shared by both splits)

Files are hard-linked when possible (no extra disk space, the 400 images are ~4 GB);
if hard links are not supported it falls back to a normal copy.
"""
import os
import random
import shutil
import sys
from pathlib import Path

# === CONFIGURATION ===
DATASET_ROOT = Path("semantic_drone_dataset")
if len(sys.argv) > 1:
    DATASET_ROOT = Path(sys.argv[1])

base_dir = DATASET_ROOT / "training_set"
image_dir = base_dir / "images"
train_dir = base_dir / "train" / "images"
valid_dir = base_dir / "valid" / "images"

split_ratio = 0.8  # 80% train, 20% valid
random.seed(42)

# === CREATE OUTPUT FOLDERS ===
train_dir.mkdir(parents=True, exist_ok=True)
valid_dir.mkdir(parents=True, exist_ok=True)

# === GET IMAGE FILES ===
images = sorted(f for f in os.listdir(image_dir)
                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff')))
random.shuffle(images)

# === SPLIT DATA ===
split_point = int(len(images) * split_ratio)
train_images = images[:split_point]
valid_images = images[split_point:]


def link_or_copy(file_list, target_dir):
    for img_file in file_list:
        src, dst = image_dir / img_file, target_dir / img_file
        if dst.exists():
            continue
        try:
            os.link(src, dst)          # hard link: instant, no extra space
        except OSError:
            shutil.copy2(src, dst)     # fallback: real copy


link_or_copy(train_images, train_dir)
link_or_copy(valid_images, valid_dir)

print(f"Done! {len(train_images)} training images and {len(valid_images)} validation images.")
print(f"Train folder: {train_dir}")
print(f"Valid folder: {valid_dir}")
