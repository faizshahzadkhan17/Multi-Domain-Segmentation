import torch
import numpy as np

from PIL import Image
from torchvision import transforms


# -------------------------------------------------------
# Image preprocessing for SegFormer
# -------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess(image):

    tensor = transform(image)

    return tensor.unsqueeze(0)


# -------------------------------------------------------
# Color map for segmentation classes
# -------------------------------------------------------

COLOR_MAP = np.array([

    [0, 0, 0],          # background
    [34, 139, 34],      # trees
    [60, 179, 113],     # lush bushes
    [189, 183, 107],    # dry grass
    [210, 180, 140],    # dry bushes
    [128, 128, 128],    # clutter
    [255, 182, 193],    # flowers
    [139, 69, 19],      # logs
    [105, 105, 105],    # rocks
    [222, 184, 135],    # landscape
    [135, 206, 235],    # sky

    # extra colors for mountain / roads classes
    [255, 99, 71],
    [255, 215, 0],
    [173, 255, 47],
    [0, 255, 255],
    [138, 43, 226],
    [255, 20, 147],
    [255, 140, 0],
    [70, 130, 180],
    [154, 205, 50],
    [255, 105, 180]

])


# -------------------------------------------------------
# Convert class mask → RGB segmentation
# -------------------------------------------------------

def decode_segmap(mask):

    h, w = mask.shape

    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls in range(len(COLOR_MAP)):

        rgb[mask == cls] = COLOR_MAP[cls]

    return Image.fromarray(rgb)


# -------------------------------------------------------
# Overlay segmentation mask
# -------------------------------------------------------

def overlay_mask(image, mask, alpha=0.5):

    image_np = np.array(image)
    mask_np = np.array(mask)

    overlay = (
        image_np * (1 - alpha) +
        mask_np * alpha
    ).astype(np.uint8)

    return Image.fromarray(overlay)