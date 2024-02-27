import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, accuracy_score
from scipy.spatial.distance import directed_hausdorff

from raac.dataset import CrackDataset
from raac.model import CrackDetectionModel
from raac.model_deep_1 import CrackDetectionModelDeep1
from torch.optim.lr_scheduler import StepLR


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, input, target):
        bce = self.bce_loss(input, target)
        dice = self.dice_loss(torch.sigmoid(input), target)
        total_loss = bce + max(dice, 0)
        return total_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1.0
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        intersection = (input_flat * target_flat).sum()
        dice_score = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice_score


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.contiguous().view(-1)
    y_pred_f = y_pred.contiguous().view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def jaccard_index(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / union


def hausdorff_distance_95th(y_true, y_pred):
    pts_true = np.argwhere(y_true)
    pts_pred = np.argwhere(y_pred)
    if not pts_true.any() or not pts_pred.any():
        return np.nan
    return np.percentile([directed_hausdorff(pts_true, pts_pred)[0], directed_hausdorff(pts_pred, pts_true)[0]], 95)


def train_one_epoch(model, dataloader, optimizer, criterion_seg, criterion_cls, device):
    model.train()
    running_loss = 0.0
    for images, masks, labels in tqdm(dataloader):
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()

        seg_output, cls_output = model(images)

        # Ensure masks match the dimensionality of seg_output
        masks = masks.squeeze(1)  # Remove channel dimension if it's 1

        seg_output = seg_output.squeeze(
            1) if seg_output.dim() > 3 else seg_output  # Conditional squeeze based on your model's output

        loss_seg = criterion_seg(seg_output, masks)  # Use combined loss for segmentation
        loss_cls = criterion_cls(cls_output, labels)  # Use BCE or another appropriate loss for classification
        loss = loss_seg + loss_cls  # Combine losses if doing both tasks
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion_seg, criterion_cls, device):
    model.eval()
    running_loss = 0.0
    total_jaccard = 0.0  # Ensure this is initialized outside and before the loop
    hausdorff_distances = []
    total_recall = total_precision = total_accuracy = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, masks, labels in dataloader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            seg_output, cls_output = model(images)

            # Ensure masks match the dimensionality of seg_output
            masks = masks.squeeze(1)  # Remove channel dimension

            seg_output = seg_output.squeeze(
                1) if seg_output.dim() > 3 else seg_output  # Conditional squeeze based on your model's output

            loss_seg = criterion_seg(seg_output, masks)
            loss_cls = criterion_cls(cls_output, labels)
            loss = loss_seg + loss_cls
            running_loss += loss.item()

            seg_output_sigmoid = torch.sigmoid(seg_output).squeeze(1) > 0.5
            cls_output_sigmoid = torch.sigmoid(cls_output) > 0.5

            for mask, prediction in zip(masks.squeeze(1).cpu().numpy(), seg_output_sigmoid.cpu().numpy()):
                total_jaccard += jaccard_index(mask, prediction)
                hausdorff_distances.append(hausdorff_distance_95th(mask, prediction))

            labels_np = labels.cpu().numpy()
            predictions_np = cls_output_sigmoid.cpu().numpy()
            total_recall += recall_score(labels_np, predictions_np, average='macro', zero_division=0)
            total_precision += precision_score(labels_np, predictions_np, average='macro', zero_division=0)
            total_accuracy += accuracy_score(labels_np, predictions_np)

            n_batches += 1

    avg_loss = running_loss / n_batches
    avg_jaccard = total_jaccard / n_batches
    avg_hd95 = np.nanmean(hausdorff_distances)
    avg_recall = total_recall / n_batches
    avg_precision = total_precision / n_batches
    avg_accuracy = total_accuracy / n_batches

    logging.info(
        f"Validation - Loss: {avg_loss:.4f}, Jaccard: {avg_jaccard:.4f}, HD95: {avg_hd95:.4f}, Recall: {avg_recall:.4f}, Precision: {avg_precision:.4f}, Accuracy: {avg_accuracy:.4f}")
    return avg_loss, avg_jaccard, avg_hd95, avg_recall, avg_precision, avg_accuracy


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Crack Detection Model')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
    parser.add_argument('--image_dir', default='dataset/train', type=str, help='directory of images')
    parser.add_argument('--mask_dir', default='dataset/train_labels', type=str, help='directory of masks')
    parser.add_argument('--labels_file', default='dataset/labels.csv', type=str, help='labels file path')
    parser.add_argument('--save_path', default='output/best_crack_detection_model.pth', type=str,
                        help='path to save best model')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'deeper1'],
                        help='Type of model to use: standard or deeper1')
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Conditionally import the model based on the argument
    if args.model_type == 'standard':
        model = CrackDetectionModel().to(device)
    elif args.model_type == 'deeper1':
        model = CrackDetectionModelDeep1().to(device)
    else:
        logging.error(f"Invalid model type selected: {args.model_type}")
        return

    criterion_seg = CombinedLoss()  # Use CombinedLoss for segmentation
    criterion_cls = nn.BCEWithLogitsLoss()  # Classification loss if needed

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
    ])

    train_dataset = CrackDataset(args.image_dir, args.mask_dir, args.labels_file, args.num_classes, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Implement validation dataset loading here
    val_dataset = CrackDataset('dataset/val', 'dataset/val_labels', 'dataset/labels_val.csv', args.num_classes,
                               transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1)

    best_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_seg, criterion_cls, device)
        train_losses.append(train_loss)
        logging.info(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}")

        # Validate after each epoch
        avg_loss, avg_jaccard, avg_hd95, avg_recall, avg_precision, avg_accuracy = validate_one_epoch(model, val_loader,
                                                                                                      criterion_seg,
                                                                                                      criterion_cls,
                                                                                                      device)
        val_losses.append(avg_loss)
        # Then, log each metric individually or format the message to include all relevant metrics
        logging.info(
            f"Epoch {epoch + 1}/{args.epochs}, Validation Loss: {avg_loss:.4f}, Jaccard: {avg_jaccard:.4f}, HD95: {avg_hd95:.4f}, Recall: {avg_recall:.4f}, Precision: {avg_precision:.4f}, Accuracy: {avg_accuracy:.4f}")

        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.save_path)
            logging.info(f"Model saved at {args.save_path}")

    plt.figure(figsize=(12, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('output/training_loss.png')
    plt.show()

    logging.info('Finished Training')


if __name__ == "__main__":
    main()
