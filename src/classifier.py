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
