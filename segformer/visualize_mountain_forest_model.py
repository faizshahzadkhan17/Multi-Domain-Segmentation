import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from dataset_desert import DesertDataset
from dataset_mountain_forest import MountainForestDataset
from dataset_roads import RoadsDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================
# SELECT MODEL
# ======================================

MODEL_NAME = "mountain_forest"

# options:
# desert
# mountain_forest
# roads


# ======================================
# CONFIG BASED ON MODEL
# ======================================

if MODEL_NAME == "desert":

    NUM_CLASSES = 6
    VAL_PATH = "../Clean_Dataset/val/desert"
    MODEL_PATH = "../checkpoints/desert_segformer_b2/best_model"
    DatasetClass = DesertDataset

elif MODEL_NAME == "mountain_forest":

    NUM_CLASSES = 15
    VAL_PATH = "../Clean_Dataset/val/mountain_forest"
    MODEL_PATH = "../checkpoints/mountain_forest_segformer_b2/best_model"
    DatasetClass = MountainForestDataset

elif MODEL_NAME == "roads":

    NUM_CLASSES = 20
    VAL_PATH = "../Clean_Dataset/val/roads"
    MODEL_PATH = "../checkpoints/roads_segformer_b2/best_model"
    DatasetClass = RoadsDataset

else:

    raise ValueError("Invalid MODEL_NAME")


SAVE_DIR = f"../results/{MODEL_NAME}_results"

os.makedirs(SAVE_DIR, exist_ok=True)


# ======================================
# DATASET
# ======================================

dataset = DatasetClass(
    VAL_PATH,
    image_size=512,
    augment=False
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)


# ======================================
# LOAD MODEL
# ======================================

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_PATH,
    num_labels=NUM_CLASSES
).to(DEVICE)

model.eval()


# ======================================
# VISUALIZATION LOOP
# ======================================

print(f"\nGenerating visualization for {MODEL_NAME}...\n")

for i, (img, mask) in enumerate(loader):

    img = img.to(DEVICE)

    with torch.no_grad():

        outputs = model(pixel_values=img)

        pred = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

    image = img.cpu().numpy()[0].transpose(1,2,0)

    gt = mask.numpy()[0]


    # ======================================
    # Overlay Prediction
    # ======================================

    overlay = image.copy()

    overlay[pred > 0] = overlay[pred > 0] * 0.5 + np.array([1,0,0]) * 0.5


    # ======================================
    # Error Map
    # ======================================

    error = (pred != gt).astype(int)


    # ======================================
    # Plot
    # ======================================

    plt.figure(figsize=(20,4))

    plt.subplot(1,5,1)
    plt.title("Input")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1,5,2)
    plt.title("Ground Truth")
    plt.imshow(gt)
    plt.axis("off")

    plt.subplot(1,5,3)
    plt.title("Prediction")
    plt.imshow(pred)
    plt.axis("off")

    plt.subplot(1,5,4)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.subplot(1,5,5)
    plt.title("Error Map")
    plt.imshow(error, cmap="hot")
    plt.axis("off")


    save_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_result_{i}.png")

    plt.savefig(save_path)

    plt.show()

    plt.close()

    print(f"Saved: {save_path}")

    if i == 20:
        break