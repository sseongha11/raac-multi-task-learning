import torch
import segmentation_models_pytorch.utils as su

X_TRAIN_DIR = 'dataset/train'
Y_TRAIN_DIR = 'dataset/train_labels'

X_VALID_DIR = 'dataset/val'
Y_VALID_DIR = 'dataset/val_labels'

X_TEST_DIR = 'dataset/test'
Y_TEST_DIR = 'dataset/test_labels'

# Get class names
CLASS_NAME = ['background', 'crack']

# Get class RGB values
CLASS_RGB_VALUES = [[0, 0, 0], [255, 255, 255]]  # select_class_rgb_values

CLASS_IDX = [0, 1]  # select_class_indices

ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = CLASS_NAME
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
