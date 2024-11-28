import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.models import densenet121
from Data.load_data import load_mri_data, train_test_split 
from tqdm import tqdm

# Define the DenseNet model
class DenseNetCNN(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetCNN, self).__init__()
        # Load pre-defined DenseNet121 with pretrained weights
        self.densenet = densenet121(pretrained=False)
        # Modify the classifier for the number of classes in your dataset
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
    
    def forward(self, x):
        return self.densenet(x)


# Hyperparameters
batch_size = 32
learning_rate = 0.001
epochs = 20
image_size = (96, 96)
num_classes = 4  # notumor, glioma, meningioma, pituitary

# Load the dataset using PyTorch-friendly code
base_path = "Project3/Data/archive"  # Adjust to your actual base path

# Load the dataset
total_dataset, labels_map = load_mri_data(base_path, batch_size=batch_size, image_size=image_size)

# Split into training and testing datasets
train_set, test_set = train_test_split(total_dataset, test_fraction=0.2, batch_size=batch_size)

# Instantiate the model
model = DenseNetCNN(num_classes=num_classes).to('cuda' if torch.cuda.is_available() else 'cpu')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(epochs):
    model.train()
    train_loss, train_correct = 0, 0

    # Use tqdm to show a progress bar for the training set
    with tqdm(train_set, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as tepoch:
        for images, labels in tepoch:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

            # Update the progress bar with the current loss and accuracy
            tepoch.set_postfix(loss=train_loss / len(tepoch), accuracy=train_correct / len(train_set.dataset))

    # Validation
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for images, labels in test_set:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

    # Calculate metrics
    train_loss /= len(train_set)
    train_acc = train_correct / len(train_set.dataset)
    val_loss /= len(test_set)
    val_acc = val_correct / len(test_set.dataset)

    # Store metrics for visualization
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    # Print epoch metrics
    print(f"Epoch [{epoch+1}/{epochs}] - "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

# Visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Training Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()
