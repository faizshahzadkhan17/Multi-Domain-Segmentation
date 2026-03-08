"""
Segmentation Training Script (BASELINE)
-------------------------------------
✔ Uses CLEAN dataset
✔ Baseline training only (NOT final training)
✔ CPU safe
✔ Step 1.1 completed correctly
"""

import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# ===================== DEVICE =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===================== CLASS MAP =====================
VALUE_MAP = {
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
N_CLASSES = len(VALUE_MAP)

def convert_mask(mask):
    mask = np.array(mask)
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for k, v in VALUE_MAP.items():
        new_mask[mask == k] = v
    return Image.fromarray(new_mask)

# ===================== DATASET =====================
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(root_dir, "Color_Images")
        self.mask_dir = os.path.join(root_dir, "Segmentation")
        self.transform = transform
        self.mask_transform = mask_transform
        self.files = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.files[idx])
        mask_path = os.path.join(self.mask_dir, self.files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = convert_mask(Image.open(mask_path))

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask) * 255

        return image, mask

# ===================== MODEL HEAD =====================
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, h, w):
        super().__init__()
        self.h = h
        self.w = w
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, 1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.h, self.w, C).permute(0, 3, 1, 2)
        return self.net(x)

# ===================== MAIN =====================
def main():
    # -------- BASIC SETTINGS (BASELINE) --------
    BATCH_SIZE = 2
    EPOCHS = 10
    LR = 1e-4

    IMG_W = int(((960 // 2) // 14) * 14)
    IMG_H = int(((540 // 2) // 14) * 14)

    TOKEN_W = IMG_W // 14
    TOKEN_H = IMG_H // 14

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # -------- STEP 1.1 : CLEAN DATASET --------
    train_dir = os.path.join(script_dir, "Clean_Dataset", "train")
    val_dir   = os.path.join(script_dir, "Clean_Dataset", "val")

    print("Train dir:", train_dir)
    print("Val dir:", val_dir)

    # -------- TRANSFORMS --------
    img_tf = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    mask_tf = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor()
    ])

    train_ds = SegmentationDataset(train_dir, img_tf, mask_tf)
    val_ds   = SegmentationDataset(val_dir, img_tf, mask_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    # -------- BACKBONE --------
    print("Loading DINOv2 backbone...")
    backbone = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14"
    ).to(device).eval()

    with torch.no_grad():
        sample, _ = next(iter(train_loader))
        sample = sample.to(device)
        feats = backbone.forward_features(sample)["x_norm_patchtokens"]

    EMBED_DIM = feats.shape[2]
    print("Embedding dim:", EMBED_DIM)

    head = SegmentationHead(
        EMBED_DIM,
        N_CLASSES,
        TOKEN_H,
        TOKEN_W
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(head.parameters(), lr=LR, momentum=0.9)

    # -------- TRAINING LOOP --------
    print("\nStarting BASELINE training...\n")

    for epoch in range(EPOCHS):
        head.train()
        epoch_loss = []

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(device)
            masks = masks.squeeze(1).long().to(device)

            with torch.no_grad():
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]

            logits = head(feats)
            logits = F.interpolate(
                logits,
                size=imgs.shape[2:],
                mode="bilinear",
                align_corners=False
            )

            loss = loss_fn(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        print(f"Epoch {epoch+1} | Train Loss: {np.mean(epoch_loss):.4f}")

    # -------- SAVE MODEL --------
    save_path = os.path.join(script_dir, "baseline_segmentation_head.pth")
    torch.save(head.state_dict(), save_path)
    print("\nBaseline training completed.")
    print("Model saved at:", save_path)

if __name__ == "__main__":
    main()
