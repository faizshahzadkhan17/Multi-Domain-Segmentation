import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ================= CONFIG =================
NUM_CLASSES = 10
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 35
LR = 3e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= DATASET =================
from multi_domain_dataset import MultiDomainDataset

train_loader = DataLoader(
    MultiDomainDataset(
        root_dir="Clean_Dataset/train",
        domains=["desert", "mountain", "forest"],
        augment=True
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# ================= MODEL =================
model = SegformerForSemanticSegmentation.from_pretrained(
    "checkpoints/best_model",  # load your desert trained model
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(DEVICE)

# ================= OPTIMIZER =================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

scaler = torch.amp.GradScaler("cuda")

# ================= DICE LOSS =================
def dice_loss(pred, target):
    pred = F.softmax(pred, dim=1)

    target = target.clamp(0, NUM_CLASSES - 1)
    target_1hot = F.one_hot(target, NUM_CLASSES).permute(0,3,1,2)

    inter = (pred * target_1hot).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target_1hot.sum(dim=(2,3))

    dice = (2 * inter + 1e-5) / (union + 1e-5)

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

        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

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

            loss = (0.6 * ce + 0.4 * dl) / GRAD_ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss += loss.item() * GRAD_ACCUM
        pbar.set_postfix(loss=f"{loss.item()*GRAD_ACCUM:.4f}")

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)

    scheduler.step()

    print(f"\nEpoch {epoch+1} | Avg Loss: {avg_loss:.6f}")

    model.save_pretrained(f"{SAVE_DIR}/epoch_{epoch+1}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        model.save_pretrained(f"{SAVE_DIR}/best_model")
        print("🔥 Best model updated")


# ================= LOSS GRAPH =================
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig(f"{SAVE_DIR}/loss_curve.png")

print("✅ Training complete")