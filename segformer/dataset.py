import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


# ==============================
# DESERT DATASET VALUE MAP
# ==============================

DESERT_VALUE_MAP = {
    0: 0,
    100: 1,
    200: 2,
    300: 3,
    500: 4,
    550: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9
}


# ==============================
# MOUNTAIN DATASET VALUE MAP
# ==============================

MOUNTAIN_VALUE_MAP = {
    0: 0,   # background
    1: 9,   # sky
    2: 8,   # mountain → landscape
    3: 2,   # vegetation → lush bushes
    4: 5,   # ground → clutter
    5: 7,   # rocks
    6: 3,   # dirt → dry grass
    7: 1,   # trees
    8: 8    # terrain → landscape
}


class SegmentationDataset(Dataset):

    def __init__(self, root_dir, image_size=512, augment=False):

        self.image_dir = os.path.join(root_dir, "Color_Images")
        self.mask_dir = os.path.join(root_dir, "Segmentation")

        self.files = sorted(os.listdir(self.image_dir))

        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.files)


    # ==============================
    # AUTO-DETECT MASK FORMAT
    # ==============================

    def convert_mask(self, mask):

        arr = np.array(mask)

        unique_vals = np.unique(arr)

        # If mask contains large numbers → desert dataset
        if unique_vals.max() > 50:
            value_map = DESERT_VALUE_MAP
        else:
            value_map = MOUNTAIN_VALUE_MAP

        out = np.zeros_like(arr, dtype=np.uint8)

        for k, v in value_map.items():
            out[arr == k] = v

        return Image.fromarray(out)


    def __getitem__(self, idx):

        img_path = os.path.join(self.image_dir, self.files[idx])
        mask_path = os.path.join(self.mask_dir, self.files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = self.convert_mask(mask)

        image = TF.resize(image, (self.image_size, self.image_size))
        mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=Image.NEAREST)

        if self.augment:
            if torch.rand(1) > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        image = TF.to_tensor(image)

        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask