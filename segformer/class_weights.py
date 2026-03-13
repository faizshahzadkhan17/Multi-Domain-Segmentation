import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_class_weights(mask_dir, num_classes):

    class_counts = np.zeros(num_classes)

    mask_files = os.listdir(mask_dir)

    print(f"\nScanning {len(mask_files)} masks...")

    for file in tqdm(mask_files):

        path = os.path.join(mask_dir, file)

        mask = np.array(Image.open(path))

        for c in range(num_classes):

            class_counts[c] += np.sum(mask == c)

    print("\nClass pixel counts:")
    print(class_counts)

    total_pixels = np.sum(class_counts)

    class_freq = class_counts / total_pixels

    weights = 1 / (np.log(1.02 + class_freq))

    weights = weights / weights.sum() * num_classes

    print("\nComputed class weights:")
    print(weights)

    return weights