# Multi-Task Learning Model for RAAC Crack Segmentation and Classification

## Overview
This project, developed by [Seongha Hwang](https://www.linkedin.com/in/seongha-hwang-a478a068/) at Loughborough University, presents a sophisticated Crack Detection Model. The model is based on a modified U-Net architecture, enhanced with dropout and batch normalization. It's designed for efficient and accurate detection and classification of cracks in surfaces or materials.

## Features
- Dual-Output Architecture: Provides both pixel-wise segmentation and image-level classification.
- U-Net Based Design: Ensures precise localization and context capture for crack detection.
- Dropout and Batch Normalization: Improves generalization and speeds up convergence.
- Flexible Input: Can be adapted to various image resolutions and sizes.

## Installation
### Prerequisites
- Python 3.x
- PyTorch
- torchvision
- PIL
- numpy
- matplotlib

### Setup
Clone the repository to your local machine:

```bash
git clone https://github.com/SeonghaLoughborough/CrackDetection.git
cd CrackDetection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
### Training the Model
To train the model, run:

```bash
python train.py --epochs [num_epochs] --lr [learning_rate] --batch_size [batch_size]
```

Adjust the parameters like num_epochs, learning_rate, and batch_size as needed.

### Performing Inference
To perform inference on a new image:

```bash
python inference.py --image_path 'path/to/image.jpg'
```
This will output the segmentation and classification results.

## Project Structure
- model.py: Contains the implementation of the Crack Detection Model.
- train.py: Script for training the model.
- inference.py: Script for performing inference using a trained model.

## Dataset

```bash
dataset/
│
├── train/          # Directory with all train
│   ├── image1.png
│   ├── image2.png
│   └── ...
│
├── train_labels/           # Directory with corresponding train_labels
│   ├── image1.png   # Mask filename matches image filename
│   ├── image2.png
│   └── ...
│
└── labels.csv       # CSV file with classification labels
    ├── filename, label
    ├── image1.png, 0  # 0 could represent 'Non-Crack'
    ├── image2.png, 1  # 1 for 'Longitudinal-or-Transverse-Crack'
    ├── image3.png, 2  # 2 for 'Mixed-Crack'  
    └── ...
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
This project was developed by [Seongha Hwang](https://www.linkedin.com/in/seongha-hwang-a478a068/) at Loughborough University. Special thanks to the faculty and research staff for their support and guidance.



