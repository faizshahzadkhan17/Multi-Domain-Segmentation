import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ================= CONFIG =================
NUM_CLASSES = 6
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 30
LR = 5e-5
IMG_SIZE = (512, 512)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "../Clean_Dataset/train"
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

IGNORE_INDEX = 255

VALUE_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    27: 4,
    39: 5
}

# ================= DATASET =================
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        self.imgs = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment

        self.img_tf = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        name = self.imgs[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name))

        img = self.img_tf(img)
        mask = mask.resize(IMG_SIZE, Image.NEAREST)
        mask_np = np.array(mask)

        new_mask = np.full(mask_np.shape, IGNORE_INDEX, dtype=np.int64)
        for k, v in VALUE_MAP.items():
            new_mask[mask_np == k] = v

        if self.augment and torch.rand(1) > 0.5:
            img = torch.flip(img, dims=[2])
            new_mask = np.fliplr(new_mask).copy()  # ✅ FIX HERE

        return img, torch.from_numpy(new_mask)

# ================= DATALOADER =================
train_loader = DataLoader(
    SegDataset(
        os.path.join(DATA_DIR, "Color_Images"),
        os.path.join(DATA_DIR, "Segmentation"),
        augment=True
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

# ================= MODEL =================
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
    use_safetensors=True
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda")

# ================= DICE LOSS =================
def dice_loss(pred, target):
    pred = F.softmax(pred, dim=1)
    target = target.clamp(0, NUM_CLASSES-1)
    target_1hot = F.one_hot(target, NUM_CLASSES).permute(0,3,1,2)

    inter = (pred * target_1hot).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target_1hot.sum(dim=(2,3))
    dice = (2*inter + 1e-5) / (union + 1e-5)
    return 1 - dice.mean()

# ================= TRAIN =================
best_loss = float("inf")
losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for step, (imgs, masks) in enumerate(pbar):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        with torch.amp.autocast("cuda"):
            out = model(pixel_values=imgs, labels=masks)
            ce = out.loss
            logits_up = F.interpolate(
               out.logits,
               size=masks.shape[-2:],
               mode="bilinear",
               align_corners=False
           )
            dl = dice_loss(logits_up, masks)

            loss = (ce + dl) / GRAD_ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss += loss.item() * GRAD_ACCUM
        pbar.set_postfix(loss=f"{loss.item()*GRAD_ACCUM:.4f}")

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)

    print(f"\nEpoch {epoch+1} | Avg Loss: {avg_loss:.6f}")

    model.save_pretrained(f"{SAVE_DIR}/epoch_{epoch+1}")
    if avg_loss < best_loss:
        best_loss = avg_loss
        model.save_pretrained(f"{SAVE_DIR}/best_model")
        print("🔥 Best model updated")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig(f"{SAVE_DIR}/loss_curve.png")

print("✅ Training complete")
