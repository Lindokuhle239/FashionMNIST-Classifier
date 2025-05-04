import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, io
import pathlib
import torchvision

#defining class names corresponding to the FashionMNIST labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#defining image processing pipeline
transform = transforms.Compose([
    transforms.ToTensor(), #converts PIL image to tensor
    transforms.Normalize((0.5), (0.5)) #normalize to [-1,1] range
])

class FashionClassifier(nn.Module):
    """Neural Network for FashionMNIST classification"""
    def __init__(self):
        super().__init__()
        #defining network layers
        self.fc1 = nn.Linear(28*28, 512) #first fully connected layer
        self.fc2 = nn.Linear(512, 10) #output layer - 10 classes
        self.dropout = nn.Dropout(p=0.25) #dropout for regularization
        
    def forward(self, x):
        """Forward pass through the network"""
        #flatten the input image
        x = x.view(-1, 28*28)
        
        #applu ReLU activation and dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        #output layer with log-softmax activation
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1) #log probabilities for NLLLoss
    
def train_model():
    """Train the FashionClassifier model"""
    #load training and test datasets
    train_data = datasets.FashionMNIST(
        root='.', #data root directory
        train=True, #load trainig set
        download=False, #using local files
        transform=transform #apply defined transforms
    )
    
