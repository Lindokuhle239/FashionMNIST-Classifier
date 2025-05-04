from torchvision import datasets, transforms

#Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#Load data
train_data = datasets.FashionMNIST('.', train=True, download=False, transform=transform)
test_data = datasets.FashionMNIST('.', train=False, download=False, transform=transform)