import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from dataset_mountain_forest import MountainForestDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================
# CONFIGURATION
# ======================================

NUM_CLASSES = 15

VAL_PATH = "../Clean_Dataset/val/mountain_forest"

MODEL_PATH = "../checkpoints/mountain_forest_segformer_b2/best_model"

BATCH_SIZE = 2


# ======================================
# DATASET
# ======================================

dataset = MountainForestDataset(
    VAL_PATH,
    image_size=512,
    augment=False
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)


# ======================================
# LOAD MODEL
# ======================================

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_PATH,
    num_labels=NUM_CLASSES
).to(DEVICE)

model.eval()


# ======================================
# CONFUSION MATRIX
# ======================================

confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))


def update_confusion_matrix(pred, label):

    pred = pred.flatten()
    label = label.flatten()

    mask = (label >= 0) & (label < NUM_CLASSES)

    hist = np.bincount(
        NUM_CLASSES * label[mask] + pred[mask],
        minlength=NUM_CLASSES ** 2
    ).reshape(NUM_CLASSES, NUM_CLASSES)

    return hist


# ======================================
# EVALUATION LOOP
# ======================================

print("\nRunning Mountain + Forest Model Evaluation...\n")

with torch.no_grad():

    for imgs, masks in tqdm(loader):

        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(pixel_values=imgs)

        preds = torch.argmax(outputs.logits, dim=1)

        preds = preds.cpu().numpy()
        masks = masks.cpu().numpy()

        for p, m in zip(preds, masks):

            confusion_matrix += update_confusion_matrix(p, m)


# ======================================
# METRIC CALCULATIONS
# ======================================

TP = np.diag(confusion_matrix)

FP = confusion_matrix.sum(axis=0) - TP

FN = confusion_matrix.sum(axis=1) - TP

pixel_accuracy = TP.sum() / confusion_matrix.sum()

class_accuracy = TP / (TP + FN + 1e-6)

mean_pixel_accuracy = np.nanmean(class_accuracy)

IoU = TP / (TP + FP + FN + 1e-6)

mIoU = np.nanmean(IoU)

precision = TP / (TP + FP + 1e-6)

recall = TP / (TP + FN + 1e-6)

f1 = 2 * precision * recall / (precision + recall + 1e-6)


# ======================================
# PRINT REPORT
# ======================================

print("\n=========== MOUNTAIN FOREST MODEL REPORT ===========")

print(f"\nPixel Accuracy: {pixel_accuracy:.4f}")
print(f"Mean Pixel Accuracy: {mean_pixel_accuracy:.4f}")
print(f"Mean IoU (mIoU): {mIoU:.4f}")

print("\nPer Class IoU")

for i in range(NUM_CLASSES):

    print(f"Class {i}: {IoU[i]:.4f}")

print("\nPrecision per class")

for i in range(NUM_CLASSES):

    print(f"Class {i}: {precision[i]:.4f}")

print("\nRecall per class")

for i in range(NUM_CLASSES):

    print(f"Class {i}: {recall[i]:.4f}")

print("\nF1 Score per class")

for i in range(NUM_CLASSES):

    print(f"Class {i}: {f1[i]:.4f}")

print("\n====================================================")


# ======================================
# SAVE REPORT
# ======================================

with open("mountain_forest_model_report.txt", "w") as f:

    f.write(f"Pixel Accuracy: {pixel_accuracy}\n")
    f.write(f"Mean Pixel Accuracy: {mean_pixel_accuracy}\n")
    f.write(f"Mean IoU: {mIoU}\n")

print("\nReport saved as mountain_forest_model_report.txt")