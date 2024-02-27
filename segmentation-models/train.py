# USAGE
# python train.py --model unet --output output/unet/
# python train.py --model unet++ --output output/unetpp/
# python train.py --model deeplabv3+ --output output/deeplabv3p/
# python train.py --model fpn --output output/fpn/

import os
from matplotlib import pyplot as plt
import pandas as pd
import torch
import argparse
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as su

from torch.utils.data import DataLoader
from tqdm import tqdm
from segmentation import config
from segmentation.dataset import CracksDataset
from segmentation.helper_functions import get_validation_augmentation, get_preprocessing, get_training_augmentation


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="unet",
                choices=["unet", "unet++", "deeplabv3+", "fpn"],
                help="name of segmentation model to train")
ap.add_argument("-o", "--output", required=True, help="path to the output directory")
args = vars(ap.parse_args())

if not os.path.exists(args["output"]):
    os.makedirs(args["output"])

MODELS = {
    "unet": smp.Unet,
    "unet++": smp.UnetPlusPlus,
    "deeplabv3+": smp.DeepLabV3Plus,
    "fpn": smp.FPN,
}

# create segmentation model with pretrained encoder
model = MODELS[args["model"]](
    encoder_name=config.ENCODER,
    encoder_weights=config.ENCODER_WEIGHTS,
    classes=len(config.CLASSES),
    activation=config.ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(config.ENCODER, config.ENCODER_WEIGHTS)

# Get train and val dataset instances
train_dataset = CracksDataset(
    config.X_TRAIN_DIR, config.Y_TRAIN_DIR,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=config.CLASS_RGB_VALUES,
)

valid_dataset = CracksDataset(
    config.X_VALID_DIR, config.Y_VALID_DIR,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=config.CLASS_RGB_VALUES,
)

# Get train and val data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=5)
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
# if os.path.exists(args["output"]):
#     model = torch.load(args["output"], map_location=config.DEVICE)

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
    # torch.save(model, os.path.join(args["output"], file_name))

    file_name_best = f'best_model.pth'
    # Save model if a better val IoU score is obtained
    if best_iou_score < valid_logs['iou_score']:
        best_iou_score = valid_logs['iou_score']
        torch.save(model, os.path.join(args["output"], file_name_best))
        print('Model saved!')

print("Evaluation on Test Data: ")
train_logs_df = pd.DataFrame(train_logs_list)
valid_logs_df = pd.DataFrame(valid_logs_list)
train_logs_df.T.to_csv(os.path.join(args["output"], 'train_logs.csv'))

plt.figure(figsize=(20, 8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw=3, label='Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw=3, label='Valid')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('IoU Score', fontsize=20)
plt.title('IoU Score Plot', fontsize=20)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig(os.path.join(args["output"], 'iou_score_plot.png'))
plt.show()

plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label='Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label='Valid')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Dice Loss', fontsize=20)
plt.title('Dice Loss Plot', fontsize=20)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig(os.path.join(args["output"], 'dice_loss_plot.png'))
plt.show()
