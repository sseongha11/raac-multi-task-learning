# USAGE
# python train.py

import os
from matplotlib import pyplot as plt
import pandas as pd
import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as su

from torch.utils.data import DataLoader
from tqdm import tqdm
from raac import config
from raac.dataset import CracksDataset
from raac.helper_functions import get_validation_augmentation, get_preprocessing, get_training_augmentation


def main():
    # create segmentation model with pretrained encoder
    model = smp.DeepLabV3Plus(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        classes=len(config.CLASSES),
        activation=config.ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(config.ENCODER, config.ENCODER_WEIGHTS)

    # Get train and val dataset instances
    train_dataset = CracksDataset(
        config.x_train_dir, config.y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=config.class_rgb_values,
    )

    valid_dataset = CracksDataset(
        config.x_valid_dir, config.y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=config.class_rgb_values,
    )

    # Get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=5)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

    # define loss function
    loss = config.LOSS

    # define metrics
    metrics = config.METRIC

    # define optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=config.INIT_LR),
    ])

    # define learning rate scheduler (not used in this NB)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    # # load best saved model checkpoint from previous commit (if present)
    # if os.path.exists(config.OUTPUT_PATH):
    #     model = torch.load(config.OUTPUT_PATH, map_location=config.DEVICE)

    train_epoch = su.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=config.DEVICE,
        verbose=True,
    )

    valid_epoch = su.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=config.DEVICE,
        verbose=True,
    )

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for e in tqdm(range(0, config.EPOCHS)):
        # Perform training & validation
        print('\nEpoch: {}'.format(e))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)
        # file_name = f'model_epoch_{i}.pth'
        # torch.save(model, os.path.join('./output', file_name))

        file_name_best = f'best_model.pth'
        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, os.path.join(config.OUTPUT_PATH, file_name_best))
            print('Model saved!')

    print("Evaluation on Test Data: ")
    train_logs_df = pd.DataFrame(train_logs_list)
    valid_logs_df = pd.DataFrame(valid_logs_list)
    train_logs_df.T.to_csv(os.path.join(config.OUTPUT_PATH, 'train_logs.csv'))

    plt.figure(figsize=(20, 8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw=3, label='Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw=3, label='Valid')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('IoU Score', fontsize=20)
    plt.title('IoU Score Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig(os.path.join(config.OUTPUT_PATH, 'iou_score_plot.png'))
    plt.show()

    plt.figure(figsize=(20, 8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label='Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label='Valid')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Dice Loss', fontsize=20)
    plt.title('Dice Loss Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig(os.path.join(config.OUTPUT_PATH, 'dice_loss_plot.png'))
    plt.show()


if __name__ == '__main__':
    main()
