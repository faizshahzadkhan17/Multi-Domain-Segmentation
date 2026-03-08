import numpy as np
from collections import Counter
import os
from PIL import Image

MASK_DIR = "Clean_Dataset/train/Segmentation"

counter = Counter()

for fname in os.listdir(MASK_DIR):
    mask_path = os.path.join(MASK_DIR, fname)
    mask = np.array(Image.open(mask_path))
    counter.update(mask.flatten().tolist())

print("\n===== CLASS DISTRIBUTION =====")
total_pixels = sum(counter.values())

for k in sorted(counter.keys()):
    percent = (counter[k] / total_pixels) * 100
    print(f"Value {k:6}: {counter[k]:12} pixels ({percent:.4f}%)")

print("Total pixels:", total_pixels)
