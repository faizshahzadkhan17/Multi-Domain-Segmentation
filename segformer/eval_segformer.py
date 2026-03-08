import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

# ============================
# CONFIG
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (512, 512)
NUM_CLASSES = 6
IGNORE_INDEX = 255

DATA_DIR = "../Clean_Dataset/val"
MODEL_DIR = "checkpoints/best_model"

# ============================
# FINAL CLASS MAP (CORRECT)
# ============================
VALUE_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    27: 4,
    39: 5
}

# ============================
# DATASET
# ============================
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted(os.listdir(img_dir))

        self.img_tf = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name))

        img = self.img_tf(img)
        mask = mask.resize(IMG_SIZE, Image.NEAREST)

        mask_np = np.array(mask)
        new_mask = np.full(mask_np.shape, IGNORE_INDEX, dtype=np.int64)

        for raw, mapped in VALUE_MAP.items():
            new_mask[mask_np == raw] = mapped

        return img, torch.from_numpy(new_mask)

# ============================
# IOU FUNCTION
# ============================
def compute_iou(pred, target):
    ious = []

    for cls in range(NUM_CLASSES):
        pred_inds = pred == cls
        target_inds = target == cls

        union = (pred_inds | target_inds).sum()
        if union == 0:
            continue

        intersection = (pred_inds & target_inds).sum()
        ious.append(intersection / union)

    return np.mean(ious) if len(ious) > 0 else np.nan

# ============================
# MAIN
# ============================
if __name__ == "__main__":

    print("🔍 Evaluating Validation Set")

    val_dataset = SegDataset(
        img_dir=os.path.join(DATA_DIR, "Color_Images"),
        mask_dir=os.path.join(DATA_DIR, "Segmentation")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_DIR,
        num_labels=NUM_CLASSES
    ).to(DEVICE)

    model.eval()
    all_ious = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="🔍 Evaluating Validation Set"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(pixel_values=images)
            logits = outputs.logits

            logits = torch.nn.functional.interpolate(
                logits,
                size=IMG_SIZE,
                mode="bilinear",
                align_corners=False
            )

            preds = torch.argmax(logits, dim=1)

            iou = compute_iou(
                preds.squeeze().cpu().numpy(),
                masks.squeeze().cpu().numpy()
            )

            if not np.isnan(iou):
                all_ious.append(iou)

    mean_iou = np.mean(all_ious)
    print("\n==============================")
    print(f"✅ Mean IoU (Validation): {mean_iou:.4f}")
    print("==============================")
