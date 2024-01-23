# USAGE
# python inference.py

import os
import random
import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as su

from torch.utils.data import DataLoader
from raac import config
from raac.dataset import CracksDataset
from raac.helper_functions import colour_code_segmentation, crop_image, get_preprocessing, get_validation_augmentation, \
    reverse_one_hot, visualize


def main():
    # load best saved model checkpoint from the current run
    if os.path.exists(config.MODEL_PATH):
        best_model = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        print('Loaded DeepLabV3+ model from this run.')
    else:
        print('No saved model checkpoint found. Exiting...')
        exit()

    # # load best saved model checkpoint from previous commit (if present)
    # elif os.path.exists('/home/sh/Documents/deep-learning/crack-detection-deeplabv3p/best_model.pth'):
    #     best_model = torch.load('/home/sh/Documents/deep-learning/crack-detection-deeplabv3p/best_model.pth', map_location=config.DEVICE)
    #     print('Loaded DeepLabV3+ model from a previous commit.')

    preprocessing_fn = smp.encoders.get_preprocessing_fn(config.ENCODER, config.ENCODER_WEIGHTS)

    # create test dataloader (with preprocessing operation: to_tensor(...))
    test_dataset = CracksDataset(
        config.x_test_dir, config.y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=config.class_rgb_values,
    )

    test_dataloader = DataLoader(test_dataset)

    # define loss function
    loss = config.LOSS

    # define metrics
    metrics = config.METRIC

    # Load best saved model checkpoint
    test_epoch = su.train.ValidEpoch(
        best_model,
        loss=loss,
        metrics=metrics,
        device=config.DEVICE,
        verbose=True,
    )

    valid_logs = test_epoch.run(test_dataloader)

    print("Evaluation on Test Data: ")
    print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
    print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")

    # Visualize predictions on test dataset

    # test dataset for visualization (without preprocessing transformations)
    test_dataset_vis = CracksDataset(
        config.x_test_dir, config.y_test_dir,
        augmentation=get_validation_augmentation(),
        class_rgb_values=config.class_rgb_values,
    )

    # get a random test image/mask index
    random_idx = random.randint(0, len(test_dataset_vis) - 1)
    image, mask = test_dataset_vis[random_idx]

    visualize(
        original_image = image,
        ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), config.class_rgb_values),
        one_hot_encoded_mask = reverse_one_hot(mask)
    )

    # visualize predictions on test dataset 
    for idx in range(len(test_dataset)):
        image, gt_mask = test_dataset[idx]
        image_vis = crop_image(test_dataset_vis[idx][0].astype('uint8'))
        x_tensor = torch.from_numpy(image).to(config.DEVICE).unsqueeze(0)
        # Predict test image
        pred_mask = best_model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        # Get prediction channel corresponding to building
        pred_building_heatmap = pred_mask[:, :, config.class_names.index('crack')]
        pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), config.class_rgb_values))
        # Convert gt_mask from `CHW` format to `HWC` format
        gt_mask = np.transpose(gt_mask, (1, 2, 0))
        gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), config.class_rgb_values))
        cv2.imwrite(os.path.join(config.OUTPUT_PATH, f"sample_pred_{idx}.png"),
                    np.hstack([image_vis, gt_mask, pred_mask])[:, :, ::-1])

        visualize(
            original_image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pred_mask,
            predicted_building_heatmap=pred_building_heatmap
        )

        if idx == 3:
            break

if __name__ == '__main__':
    main()
