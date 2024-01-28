import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from raac.dataset import CrackDataset
from raac.model import CrackDetectionModel
from torch.optim.lr_scheduler import StepLR


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, masks, labels in dataloader:
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()
        seg_output, cls_output = model(images)
        masks = masks.squeeze(1)
        seg_output = seg_output.squeeze(1)
        loss = criterion(cls_output, labels) + criterion(seg_output, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks, labels in dataloader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            seg_output, cls_output = model(images)
            masks = masks.squeeze(1)
            seg_output = seg_output.squeeze(1)
            loss = criterion(cls_output, labels) + criterion(seg_output, masks)
            running_loss += loss.item()
    return running_loss / len(dataloader)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Crack Detection Model')
    parser.add_argument('--epochs', default=25, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
    parser.add_argument('--image_dir', default='dataset/images', type=str, help='directory of images')
    parser.add_argument('--mask_dir', default='dataset/masks', type=str, help='directory of masks')
    parser.add_argument('--labels_file', default='dataset/labels.csv', type=str, help='labels file path')
    parser.add_argument('--save_path', default='output/best_crack_detection_model.pth', type=str,
                        help='path to save best model')
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = CrackDetectionModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    train_dataset = CrackDataset(args.image_dir, args.mask_dir, args.labels_file, args.num_classes, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Implement validation dataset loading here
    # val_dataset = ...
    # val_loader = DataLoader(val_dataset, ...)

    best_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        logging.info(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}")

        # Validate after each epoch
        # val_loss = validate_one_epoch(model, val_loader, criterion, device)
        # val_losses.append(val_loss)
        # logging.info(f"Epoch {epoch + 1}/{args.epochs}, Validation Loss: {val_loss:.4f}")

        scheduler.step()

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), args.save_path)
            logging.info(f"Model saved at {args.save_path}")

    plt.figure(figsize=(12, 4))
    plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('output/training_loss.png')
    plt.show()

    logging.info('Finished Training')


if __name__ == "__main__":
    main()
