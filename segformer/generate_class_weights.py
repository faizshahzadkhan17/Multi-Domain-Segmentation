import numpy as np
from class_weights import compute_class_weights


# ==========================================
# DESERT
# ==========================================

desert_weights = compute_class_weights(
    "../Clean_Dataset/train/desert/Segmentation",
    num_classes=6
)

np.save("weights_desert.npy", desert_weights)


# ==========================================
# MOUNTAIN + FOREST
# ==========================================

mountain_weights = compute_class_weights(
    "../Clean_Dataset/train/mountain_forest/Segmentation",
    num_classes=15
)

np.save("weights_mountain.npy", mountain_weights)


# ==========================================
# ROADS
# ==========================================

roads_weights = compute_class_weights(
    "../Clean_Dataset/train/roads/Segmentation",
    num_classes=20
)

np.save("weights_roads.npy", roads_weights)

print("\nAll weights generated and saved.")