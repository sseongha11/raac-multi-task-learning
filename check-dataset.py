# USAGE
# python check-dataset.py

import random
from raac.dataset import CracksDataset
from raac import config
from raac.helper_functions import colour_code_segmentation, reverse_one_hot, visualize


dataset = CracksDataset(config.x_train_dir, config.y_train_dir, class_rgb_values=config.class_rgb_values)
random_idx = random.randint(0, len(dataset)-1)
image, mask = dataset[2]

visualize(
    original_image = image,
    ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), config.class_rgb_values),
    one_hot_encoded_mask = reverse_one_hot(mask)
)