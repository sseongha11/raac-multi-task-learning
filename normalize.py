import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the train.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


# Usage
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    # ... other transformations ...
])

dataset = CustomImageDataset(csv_file='dataset/labels.csv', img_dir='dataset/images', transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize sum, sum of squares, and count variables
sum_rgb = torch.tensor([0.0, 0.0, 0.0])
sum_squares_rgb = torch.tensor([0.0, 0.0, 0.0])
count = 0

# Iterate over the dataset
for data, _ in tqdm(loader):
    # data is a batch of train with shape [B, C, H, W]
    sum_rgb += data.sum(dim=[0, 2, 3])  # sum over batch, height, and width dimensions
    sum_squares_rgb += (data ** 2).sum(dim=[0, 2, 3])  # sum of squares
    count += data.numel() / data.size(1)  # count pixels

# Calculate mean for each channel
mean_rgb = sum_rgb / count

# Calculate standard deviation for each channel
std_rgb = (sum_squares_rgb / count - mean_rgb ** 2) ** 0.5

print("Mean for each channel:", mean_rgb)
print("Standard deviation for each channel:", std_rgb)
