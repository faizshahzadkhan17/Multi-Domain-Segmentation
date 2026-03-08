import os
import cv2
import shutil
import numpy as np

# ---------------------------------------------------------
# Auto-detect project base directory
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# Auto-detect inner project folder (handles nested folders)
# ---------------------------------------------------------
PROJECT_FOLDER = None
for item in os.listdir(BASE_DIR):
    if os.path.isdir(os.path.join(BASE_DIR, item)) and "Segment Project" in item:
        PROJECT_FOLDER = os.path.join(BASE_DIR, item)
        break

if PROJECT_FOLDER is None:
    raise FileNotFoundError("Could not find inner 'Desert Segment Project' folder")

# ---------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------
SRC_DATASET = os.path.join(
    PROJECT_FOLDER,
    "Offroad_Segmentation_Training_Dataset"
)

DST_DATASET = os.path.join(
    PROJECT_FOLDER,
    "Clean_Dataset"
)

SPLITS = ["train", "val"]

# ---------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------
def is_valid_image(path):
    img = cv2.imread(path)
    return img is not None

def is_valid_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return mask is not None and np.unique(mask).size > 1

# ---------------------------------------------------------
# Cleaning logic
# ---------------------------------------------------------
def clean_split(split):
    src_img_dir = os.path.join(SRC_DATASET, split, "Color_Images")
    src_mask_dir = os.path.join(SRC_DATASET, split, "Segmentation")

    if not os.path.exists(src_img_dir):
        raise FileNotFoundError(f"Missing image folder: {src_img_dir}")
    if not os.path.exists(src_mask_dir):
        raise FileNotFoundError(f"Missing mask folder: {src_mask_dir}")

    dst_img_dir = os.path.join(DST_DATASET, split, "Color_Images")
    dst_mask_dir = os.path.join(DST_DATASET, split, "Segmentation")

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)

    images = sorted(os.listdir(src_img_dir))
    masks = sorted(os.listdir(src_mask_dir))

    copied = 0

    for img_name, mask_name in zip(images, masks):
        img_path = os.path.join(src_img_dir, img_name)
        mask_path = os.path.join(src_mask_dir, mask_name)

        if not is_valid_image(img_path):
            continue
        if not is_valid_mask(mask_path):
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img.shape[:2] != mask.shape:
            mask = cv2.resize(
                mask,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        shutil.copy(img_path, os.path.join(dst_img_dir, img_name))
        cv2.imwrite(os.path.join(dst_mask_dir, mask_name), mask)

        copied += 1

    print(f"{split}: copied {copied} clean samples")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Source dataset:", SRC_DATASET)
    print("Destination dataset:", DST_DATASET)
    print("Starting safe data cleaning...\n")

    for split in SPLITS:
        clean_split(split)

    print("\n✅ Data cleaning completed successfully")
