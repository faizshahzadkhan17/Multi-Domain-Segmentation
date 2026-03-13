import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from ohem_loss import OHEMLoss

torch.backends.cudnn.benchmark = True

from class_weights import compute_class_weights
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from dataset_desert import DesertSegmentationDataset


# ======================================================
# CONFIGURATION
# ======================================================

NUM_CLASSES = 6
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 45
LR = 3e-5
IMAGE_SIZE = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_PATH = "../Clean_Dataset/train/desert"
VAL_PATH = "../Clean_Dataset/val/desert"

CHECKPOINT_DIR = "../checkpoints/desert_segformer_b2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ======================================================
# DATASETS
# ======================================================

train_dataset = DesertSegmentationDataset(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    augment=True
)

val_dataset = DesertSegmentationDataset(
    VAL_PATH,
    image_size=IMAGE_SIZE,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)


# ======================================================
# MODEL
# ======================================================

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(DEVICE)
# Prevent overfitting
model.config.hidden_dropout_prob = 0.1
model.config.attention_probs_dropout_prob = 0.1

# ======================================================
# COMPUTE CLASS WEIGHTS
# ======================================================

mask_dir = "../Clean_Dataset/train/desert/Segmentation"

weights = compute_class_weights(mask_dir, NUM_CLASSES)

class_weights = torch.tensor(weights).float().to(DEVICE)

criterion = OHEMLoss()

# ======================================================
# OPTIMIZER + SCHEDULER
# ======================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)

scaler = torch.amp.GradScaler("cuda")


# ======================================================
# DICE LOSS
# ======================================================

def dice_loss(pred, target):

    pred = F.softmax(pred, dim=1)

    target = target.clamp(0, NUM_CLASSES - 1)

    target_1hot = F.one_hot(target, NUM_CLASSES).permute(0,3,1,2).float()

    inter = (pred * target_1hot).sum(dim=(2,3))

    union = pred.sum(dim=(2,3)) + target_1hot.sum(dim=(2,3))

    dice = (2 * inter + 1e-5) / (union + 1e-5)

    return 1 - dice.mean()


# ======================================================
# mIoU FUNCTION
# ======================================================

def compute_miou(pred, mask):

    pred = torch.argmax(pred, dim=1)

    ious = []

    for cls in range(NUM_CLASSES):

        pred_inds = (pred == cls)
        target_inds = (mask == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    if len(ious) == 0:
        return 0

    return sum(ious) / len(ious)


# ======================================================
# VALIDATION
# ======================================================

def validate():

    model.eval()

    total_loss = 0
    total_miou = 0

    with torch.no_grad():

        for imgs, masks in val_loader:

            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            with torch.amp.autocast("cuda"):

                out = model(pixel_values=imgs)


                logits = F.interpolate(
                    out.logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

                ce = criterion(logits, masks)

                dl = dice_loss(logits, masks)

                loss = (0.6 * ce + 0.4 * dl) 

            miou = compute_miou(logits, masks)

            total_loss += loss.item()
            total_miou += miou

    return total_loss / len(val_loader), total_miou / len(val_loader)


# ======================================================
# TRAIN LOOP
# ======================================================

best_miou = 0

# Early stopping settings
patience = 6
no_improve_epochs = 0

train_losses = []
val_losses = []
val_mious = []

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for step, (imgs, masks) in enumerate(pbar):

        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        with torch.amp.autocast("cuda"):

            out = model(pixel_values=imgs)

            logits = F.interpolate(
                out.logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            ce = criterion(logits, masks)

            dl = dice_loss(logits, masks)

            loss = (0.6 * ce + 0.4 * dl) / GRAD_ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        epoch_loss += loss.item() * GRAD_ACCUM

        pbar.set_postfix(loss=loss.item() * GRAD_ACCUM)

    avg_train_loss = epoch_loss / len(train_loader)

    val_loss, val_miou = validate()

    scheduler.step()

    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    val_mious.append(val_miou)

    print(f"\nTrain Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val mIoU: {val_miou:.4f}")
    print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

    model.save_pretrained(f"{CHECKPOINT_DIR}/epoch_{epoch+1}")

    if val_miou > best_miou:

        best_miou = val_miou
        no_improve_epochs = 0

        model.save_pretrained(f"{CHECKPOINT_DIR}/best_model")

        print("🔥 Best model updated")

    else:

        no_improve_epochs += 1

        print(f"No improvement for {no_improve_epochs} epochs")

        if no_improve_epochs >= patience:

            print("🛑 Early stopping triggered")

            break


# ======================================================
# SAVE TRAINING CURVE
# ======================================================

plt.figure()

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Desert Training Curve")

plt.savefig(f"{CHECKPOINT_DIR}/training_curve.png")

print("✅ Desert training complete")