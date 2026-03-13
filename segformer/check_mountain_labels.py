import os
import numpy as np
from PIL import Image

# ===============================
# CHANGE THIS PATH IF NEEDED
# ===============================
MASK_FOLDER = r"Clean_Dataset/train/mountain/Segmentation"

# ===============================
# CHECK IF FOLDER EXISTS
# ===============================
if not os.path.exists(MASK_FOLDER):
    print("\n❌ Folder not found:")
    print(MASK_FOLDER)
    exit()

print("\nScanning masks in:")
print(MASK_FOLDER)

unique_values = set()
count = 0

# ===============================
# SCAN ALL MASKS
# ===============================
for file in os.listdir(MASK_FOLDER):

    if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):

        path = os.path.join(MASK_FOLDER, file)

        img = Image.open(path)
        arr = np.array(img)

        vals = np.unique(arr)

        unique_values.update(vals)

        count += 1

        if count % 100 == 0:
            print(f"Checked {count} images...")

# ===============================
# RESULTS
# ===============================
print("\n==============================")
print("Total masks scanned:", count)
print("Unique mask values found:")
print(sorted(unique_values))
print("==============================\n")