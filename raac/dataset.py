import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F


class CrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, labels_file, num_classes, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.labels = pd.read_csv(labels_file)
        self.labels.set_index("filename", inplace=True)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx].name
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        label = self.labels.iloc[idx].label
        # Convert label to one-hot encoding
        one_hot_label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, one_hot_label.float()


# Example Usage
image_dir = 'dataset/images'
mask_dir = 'dataset/masks'
labels_file = 'dataset/labels.csv'

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

num_classes = 3  # Update this based on your number of classes
dataset = CrackDataset(image_dir, mask_dir, labels_file, num_classes, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
