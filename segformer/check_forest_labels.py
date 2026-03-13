import os
import numpy as np
from PIL import Image

MASK_FOLDER = r"Clean_Dataset/train/forest/Segmentation"

if not os.path.exists(MASK_FOLDER):
    print("❌ Folder not found:", MASK_FOLDER)
    exit()

values = set()
count = 0

for file in os.listdir(MASK_FOLDER):

    if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):

        path = os.path.join(MASK_FOLDER, file)

        img = Image.open(path)
        arr = np.array(img)

        vals = np.unique(arr)
        values.update(vals)

        count += 1

print("\n==============================")
print("Total masks scanned:", count)
print("Unique mask values found:")
print(sorted(values))
print("==============================")