import os
import numpy as np
from PIL import Image

ROOT = "Clean_Dataset/train"

for domain in os.listdir(ROOT):

    seg_path = os.path.join(ROOT, domain, "Segmentation")

    if not os.path.exists(seg_path):
        continue

    values = set()

    for file in os.listdir(seg_path):
        if file.endswith(".png") or file.endswith(".jpg"):
            img = Image.open(os.path.join(seg_path, file))
            arr = np.array(img)
            values.update(np.unique(arr))

    print("\nDATASET:", domain)
    print("Unique values:", sorted(values))