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
    
    test_data = datasets.FashionMNIST(
        root='.',
        train=False, #load test set
        download=False,
        transform=transform
    )
    
    #data loaders for batch processing
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    #initialize model, loss function, and optimizer
    model = FashionClassifier()
    criterion = nn.NLLLoss() #negative log likelihood loss
    optimizer = optim.Adam(model.parameters(), lr=0.001) #Adam optimizer
    
    #training loop
    epochs = 15
    for epoch in range(epochs):
        running_loss = 0.0 #track loss per epoch
        
        #batch training
        for images, labels in train_loader:
            #zero parameter gradients
            optimizer.zero_grad()
            
            #forwar pass
            output = model(images)
            
            #calc loss
            loss = criterion(output, labels)
            
            #backward pass and optimizer
            loss.backward()
            optimizer.step()
            
            #accumulate loss
            running_loss += loss.item()
            
        #print epoch stats
        avg_loss = running_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
    #save training model weights
    torch.save(model.state_dict(), 'fashion_classifier.pth')
    return model

def classify_image(model, image_path):
    """Classify a single image using the trained model"""
    #error handling for image loading
    try:
        img = torchvision.io.read_image(image_path, model=torchvision.io.ImageReadMode.GRAY)
    except:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    #read and process the image
    img = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.GRAY)
    img = img.float() / 255.0 #normalize to [0,1] range
    
    #apply same transforms as training data
    transform = transforms.Compose([
        transforms.Resize((28,28)), #ensure correct size
        transforms.Normalize((0.5), (0.5))
    ])
    img = transform(img)
    
    #add batch dimension and predict
    with torch.no_grad(): #disable gradient calc
        img = img.unsqueeze(0) #add batch dimension (1,1,28,28)
        output = model(img)
        _, predicted = torch.max(output, 1) #get predicted class
        
    return class_names[predicted.item()] #return class name

if __name__ == "__main__":
    print("Trainig Model...🔃")
    model = train_model()
    
    #evaluate model on test set
    test_data = datasets.FashionMNIST(root='.', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    correct = 0
    total = 0
    with torch.no_grad(): #disable gradients for evaluation
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"\nTest Accuracy: {100 * correct / total:.2f}%")
    
    #interactive classification loop
    print("\n====================================================================")
    print("\nReady for classification‼️")
    while True:
        path = input("Please enter a filepath (or 'exit' to quit): ").strip()
        if path.lower() == 'exit':
            break
        if not pathlib.Path(path).exists():
            print("File not found!!🥲")
            continue
        prediction = classify_image(model, path)
        print(f"Classifier: {prediction}")
        
    print("Exiting...👋")
        
    
    

