import torch
import matplotlib.pyplot as plt

from dataset_desert import DesertSegmentationDataset
from dataset_mountain_forest import MountainForestDataset
from dataset_roads import RoadsDataset


def check_dataset(dataset, name):

    print(f"\nChecking {name} dataset")

    img, mask = dataset[0]

    print("Image tensor shape:", img.shape)
    print("Mask tensor shape:", mask.shape)

    print("Unique mask classes:", torch.unique(mask))

    # convert image tensor to numpy for display
    img_np = img.permute(1,2,0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title("Image")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Mask")
    plt.imshow(mask.numpy())
    plt.axis("off")

    plt.show()


if __name__ == "__main__":

    desert = DesertSegmentationDataset("../Clean_Dataset/train/desert")
    mountain = MountainForestDataset("../Clean_Dataset/train/mountain_forest")
    roads = RoadsDataset("../Clean_Dataset/train/roads")

    check_dataset(desert, "DESERT")
    check_dataset(mountain, "MOUNTAIN_FOREST")
    check_dataset(roads, "ROADS")