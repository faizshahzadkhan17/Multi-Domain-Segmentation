import os
from PIL import Image

ROOT = "Clean_Dataset/train"

for dataset in os.listdir(ROOT):

    train_img = os.path.join(ROOT, dataset, "Color_Images")

    if not os.path.exists(train_img):
        continue

    sizes = set()

    for f in os.listdir(train_img)[:200]:
        img = Image.open(os.path.join(train_img, f))
        sizes.add(img.size)

    print(dataset, "example sizes:", sizes)