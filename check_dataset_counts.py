import os

ROOT = "Clean_Dataset"

for split in ["train", "val"]:

    split_path = os.path.join(ROOT, split)
    print("\nSPLIT:", split)

    for domain in os.listdir(split_path):

        domain_path = os.path.join(split_path, domain)

        if not os.path.isdir(domain_path):
            continue

        img_dir = os.path.join(domain_path, "Color_Images")
        mask_dir = os.path.join(domain_path, "Segmentation")

        print("DOMAIN:", domain)

        if os.path.exists(img_dir):
            print("images:", len(os.listdir(img_dir)))

        if os.path.exists(mask_dir):
            print("masks:", len(os.listdir(mask_dir)))
            