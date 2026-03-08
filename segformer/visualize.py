import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
import torchvision.transforms.functional as TF

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 6
IMG_SIZE = 512

VAL_IMG_DIR = "../Clean_Dataset/val/Color_Images"
VAL_MASK_DIR = "../Clean_Dataset/val/Segmentation"

MODEL_PATH = "checkpoints/best_model"
SAVE_DIR = "segformer_eval/colored_visuals"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# CLASS MAP
# =========================
VALUE_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    27: 4,
    39: 5
}

# =========================
# COLORS (BGR → RGB)
# =========================
COLORS = {
    0: (0, 0, 0),         # background
    1: (255, 0, 0),       # red
    2: (0, 255, 0),       # green
    3: (0, 0, 255),       # blue
    4: (255, 255, 0),     # yellow
    5: (255, 0, 255)      # magenta
}

# =========================
# UTILS
# =========================
def map_mask(mask):
    mapped = np.zeros_like(mask)
    for k, v in VALUE_MAP.items():
        mapped[mask == k] = v
    return mapped

def colorize(mask):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        colored[mask == cls] = color
    return colored

# =========================
# LOAD MODEL
# =========================
print("🔹 Loading model...")
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_PATH,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(DEVICE)
model.eval()

# =========================
# PROCESS IMAGES
# =========================
images = sorted(os.listdir(VAL_IMG_DIR))

print("🎨 Generating colored visualizations...")
for name in tqdm(images[:30]):  # first 30 samples
    img_path = os.path.join(VAL_IMG_DIR, name)
    mask_path = os.path.join(VAL_MASK_DIR, name)

    # Load image
    image = Image.open(img_path).convert("RGB")
    image = TF.resize(image, (IMG_SIZE, IMG_SIZE))
    img_tensor = TF.to_tensor(image).unsqueeze(0).to(DEVICE)

    # Load GT mask
    gt = Image.open(mask_path)
    gt = TF.resize(gt, (IMG_SIZE, IMG_SIZE), interpolation=Image.NEAREST)
    gt = map_mask(np.array(gt))
    gt_colored = colorize(gt)

    # Predict
    with torch.no_grad():
        out = model(pixel_values=img_tensor)
        logits = torch.nn.functional.interpolate(
            out.logits,
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False
        )
        pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    pred_colored = colorize(pred)

    # Stack visuals
    combined = np.hstack([
        np.array(image),
        gt_colored,
        pred_colored
    ])

    Image.fromarray(combined).save(os.path.join(SAVE_DIR, name))

print("✅ Colored visualizations saved to:", SAVE_DIR)
