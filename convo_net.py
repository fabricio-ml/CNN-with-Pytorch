# Import necessary PyTorch modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor
])

# Load MNIST training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# Create data loader for training set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Load MNIST test dataset
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# Create data loader for test set
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # First fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Second fully connected layer (output layer)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Apply first conv layer, ReLU activation, and pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second conv layer, ReLU activation, and pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor
        x = x.view(-1, 64 * 7 * 7)
        # Apply first fully connected layer with ReLU
        x = F.relu(self.fc1(x))
        # Apply second fully connected layer (output layer)
        x = self.fc2(x)
        return x

# Create an instance of the CNN
net = CNN()

# Define loss function
criterion = nn.CrossEntropyLoss()
# Define optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Set number of training epochs
num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get inputs and labels from data loader
        inputs, labels = data
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = net(inputs)
        # Compute loss
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        # Accumulate loss
        running_loss += loss.item()
        # Print statistics every 100 mini-batches
        if i % 100 == 99:    
            print(f'Epoch [{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# Evaluate the model on test set
correct = 0
total = 0
with torch.no_grad():  # Disable gradient computation
    for data in testloader:
        # Get images and labels from data loader
        images, labels = data
        # Forward pass
        outputs = net(images)
        # Get predicted class
        _, predicted = torch.max(outputs.data, 1)
        # Update total number of labels
        total += labels.size(0)
        # Update number of correct predictions
        correct += (predicted == labels).sum().item()

# Print accuracy
print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
