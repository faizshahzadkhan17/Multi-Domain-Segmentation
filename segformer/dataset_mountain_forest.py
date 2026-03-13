import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


# ==========================================
# Dataset configuration
# ==========================================

NUM_CLASSES = 15


class MountainForestDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        image_size: int = 512,
        augment: bool = False
    ):

        self.image_dir = os.path.join(root_dir, "Color_Images")
        self.mask_dir = os.path.join(root_dir, "Segmentation")

        self.files = sorted(os.listdir(self.image_dir))

        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.files)


    # ==========================================
    # Validate mask labels
    # ==========================================

    def validate_mask(self, mask):

        arr = np.array(mask)

        if arr.max() >= NUM_CLASSES:
            raise ValueError(
                f"Mask contains invalid label {arr.max()} "
                f"(expected < {NUM_CLASSES})"
            )

        return Image.fromarray(arr.astype(np.uint8))


    # ==========================================
    # Load one sample
    # ==========================================

    def __getitem__(self, idx):

        filename = self.files[idx]

        img_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = self.validate_mask(mask)

        # Resize
        image = TF.resize(image, (self.image_size, self.image_size))
        mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=Image.NEAREST)

        # ==========================================
        # Augmentation
        # ==========================================

        if self.augment:

            if torch.rand(1) > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if torch.rand(1) > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # small brightness variation
            if torch.rand(1) > 0.7:
                image = TF.adjust_brightness(image, 1.2)

        # Convert to tensor
        image = TF.to_tensor(image)

        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask