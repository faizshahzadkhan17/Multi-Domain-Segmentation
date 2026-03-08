import torch
import numpy as np
from torchvision import transforms
from PIL import Image

IMG_SIZE = 512

# 6-class color map
COLORS = {
    0: (0, 0, 0),        # background
    1: (255, 0, 0),      # class 1
    2: (0, 255, 0),      # class 2
    3: (0, 0, 255),      # class 3
    4: (255, 255, 0),    # class 4
    5: (255, 0, 255)     # class 5
}

def preprocess(img):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return tf(img).unsqueeze(0)

def decode_segmap(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, rgb in COLORS.items():
        color[mask == cls] = rgb
    return Image.fromarray(color)

def overlay_mask(image, mask, alpha=0.5):
    return Image.blend(image, mask, alpha)
