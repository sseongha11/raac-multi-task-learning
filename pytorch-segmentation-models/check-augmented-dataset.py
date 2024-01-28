# USAGE
# python check-augmented-dataset.py

import random
from raac.dataset import CracksDataset
from raac import config
from raac.helper_functions import colour_code_segmentation, get_training_augmentation, reverse_one_hot, visualize

augmented_dataset = CracksDataset(
    config.X_TRAIN_DIR, config.Y_TRAIN_DIR,
    augmentation=get_training_augmentation(),
    class_rgb_values=config.CLASS_RGB_VALUES,
)

random_idx = random.randint(0, len(augmented_dataset) - 1)

# Different augmentations on a random image/mask pair (256*256 crop)
for i in range(3):
    image, mask = augmented_dataset[random_idx]
    visualize(
        original_image=image,
        ground_truth_mask=colour_code_segmentation(reverse_one_hot(mask), config.CLASS_RGB_VALUES),
        one_hot_encoded_mask=reverse_one_hot(mask)
    )
