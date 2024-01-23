import torch
import os
import segmentation_models_pytorch.utils as su

BASE_PATH = '/home/sh/Documents/deep-learning/raac/crack-detection-deeplabv3p-code/'
OUTPUT_PATH = os.path.join(BASE_PATH, 'output')
MODEL_PATH = os.path.join(OUTPUT_PATH, 'best_model.pth')
DATA_DIR = os.path.join(BASE_PATH, 'dataset')

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_labels')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'test_labels')

# Get class names
class_names = ['background', 'crack']

# Get class RGB values
class_rgb_values = [[0, 0, 0], [255, 255, 255]]  # select_class_rgb_values

class_idx = [0, 1]  # select_class_indices

ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = class_names
ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation

# Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
TRAINING = True

# Set num of epochs
EPOCHS = 80
INIT_LR = 1e-4
TH = 0.5

# define loss function
LOSS = su.losses.DiceLoss()

# define metrics
METRIC = [
    su.metrics.IoU(threshold=TH),
]

# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
