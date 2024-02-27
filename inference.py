# Description: Perform inference on a single image using the trained model.
# USAGE
# python inference.py --image_path dataset/test/crack_0052.png --save_path output/overlay_image.png

import argparse
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import logging
from raac.model import CrackDetectionModel
from raac.model_deep_1 import CrackDetectionModelDeep1

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(model_path, device):
    """
    Load the trained model from a given path.
    """
    try:
        model = CrackDetectionModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def preprocess_image(image_path, transform, device):
    """
    Preprocess the image: load image, apply transformations, and move to device.
    """
    try:
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(device)
        return image_tensor
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise


def perform_inference(model, image_tensor):
    """
    Perform inference using the model and return the outputs.
    """
    with torch.no_grad():
        seg_output, cls_output = model(image_tensor)
    return seg_output, cls_output


def overlay_segmentation(original_image, seg_mask):
    """
    Overlay segmentation mask on the original image.
    """
    seg_mask_image = Image.fromarray(seg_mask.astype(np.uint8) * 255)
    return Image.blend(original_image.convert("RGBA"), seg_mask_image.convert("RGBA"), alpha=0.5)


def visualize_result(image_path, seg_output, cls_output, class_labels, seg_threshold, save_path):
    """
    Visualize the inference result by overlaying segmentation mask, showing class label, and saving the segmentation output as a PNG file.
    """
    original_image = Image.open(image_path)
    seg_output = seg_output.squeeze().cpu().numpy()
    seg_mask = seg_output > seg_threshold
    overlay_image = overlay_segmentation(original_image, seg_mask)

    # Draw class and probability
    draw = ImageDraw.Draw(overlay_image)
    try:
        font = ImageFont.truetype("resources/fonts/arial.ttf", size=20)
    except IOError:
        font = ImageFont.load_default()
        logging.warning("Arial fonts not available, using default fonts.")

    class_probabilities = torch.sigmoid(cls_output)
    class_probs_numpy = class_probabilities.squeeze().cpu().numpy()
    best_class_index = np.argmax(class_probs_numpy)
    best_class = class_labels[best_class_index]
    best_class_prob = class_probs_numpy[best_class_index]

    text = f"{best_class} ({best_class_prob:.2f})"
    draw.text((10, 10), text, fill="red", font=font)

    plt.imshow(overlay_image)
    plt.axis('off')

    if save_path:
        overlay_path = save_path
        # Ensure the overlay image is saved with '_overlay' suffix
        if not overlay_path.lower().endswith('.png'):
            overlay_path = overlay_path.rsplit('.', 1)[0] + '_overlay.png'
        overlay_image.save(overlay_path, format='PNG')
        logging.info(f"Overlay image saved to {overlay_path}")

        # Save the segmentation mask as a separate PNG file
        seg_mask_path = save_path.rsplit('.', 1)[0] + '_seg_mask.png'
        seg_mask_image = Image.fromarray((seg_mask * 255).astype(np.uint8))
        seg_mask_image.save(seg_mask_path, format='PNG')
        logging.info(f"Segmentation mask saved to {seg_mask_path}")

    plt.show()


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Crack Detection Inference')
    parser.add_argument('--model_path', type=str, default='output/best_crack_detection_model.pth',
                        help='Path to the model file')
    parser.add_argument('--image_path', type=str, default='dataset/test/crack_0052.png', help='Path to the image file')
    parser.add_argument('--seg_threshold', type=float, default=0.5, help='Segmentation threshold')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the overlay image')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'deeper1'],
                        help='Type of model to use: standard or deeper1')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_labels = ["Non-Crack", "Longitudinal-or-Transverse-Crack", "Mixed-Crack"]

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Conditionally import the model based on the argument
    if args.model_type == 'standard':
        model = CrackDetectionModel().to(device)
    elif args.model_type == 'deeper1':
        model = CrackDetectionModelDeep1().to(device)
    else:
        logging.error(f"Invalid model type selected: {args.model_type}")
        exit(1)

    image_tensor = preprocess_image(args.image_path, transform, device)
    seg_output, cls_output = perform_inference(model, image_tensor)
    visualize_result(args.image_path, seg_output, cls_output, class_labels, args.seg_threshold, args.save_path)
