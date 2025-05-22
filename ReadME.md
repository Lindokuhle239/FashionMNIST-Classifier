# FashionMNIST Classifier

## Project Overview
This repository contains an Artificial Neural Network (ANN) implementation for classifying FashionMNIST images into 10 fashion categories. The system includes model training, evaluation, and interactive image classification capabilities.

## File Structure
.
├── fashion-jpegs/ # Sample test images
├── FashionMNIST/ # Dataset directory
├── src/
│ ├── classifier.py # Main training and classification script
│ └── data_loading.py # Data loading and preprocessing
├── .gitignore
├── log.txt # Training process log
└── README.md # This documentation


## Key Features
- 88-89% test accuracy on FashionMNIST
- Interactive image classification interface
- Preprocessing pipeline for JPEG conversion
- Detailed training logs

## File Descriptions

### 1. `classifier.py`
**Purpose**: Main executable script  
**Functionality**:
- Defines neural network architecture (`FashionClassifier`)
- Implements training loop with Adam optimizer
- Evaluates model on test set
- Provides interactive image classification
- Measures and logs training times

**Usage**:
```bash
python3 src/classifier.py

### 2. data_loading.py
**Purpose**: Data management
**Functionality**:
- Loads FashionMNIST dataset
- Applies transformations:
    - ToTensor() conversion
    - Normalization (μ=0.5, σ=0.5)
- Creates DataLoader instances for batch processing

### 3. Pre-trained Model
**File**: fashion_classifier.pth
**Contains**:
- Serialized model weights (generated after training)
- Best performing model state_dict

### 4. log.txt
**Purpose**: Training documentation
**Contains**:
- Epoch-wise loss values
- Final test accuracy (typically 88-89%)
- Training duration measurements

### Usage Example
** 1.Train the model**:
```bash
python3 src/classifier.py

** 2. Classify images interactively***:
```bash
> Please enter a filepath: fashion-jpegs/bag.jpg
Classifier: Bag

### Dependencies
- Python 3.10+
- PyTorch 2.1.2
- TorchVision 0.16.2
- NumPy (<2.0)
