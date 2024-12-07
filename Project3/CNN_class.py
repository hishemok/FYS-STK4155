import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.models import densenet121, resnet50, efficientnet_b0, vit_b_16
from Data.load_data import load_mri_data, train_test_split 
from tqdm import tqdm


class CNN:
    def __init__(self, train, validation, test, num_classes,model,labels_map):
        """
        Parameters:
        Training dataset (torch.utils.data.Dataset): The training dataset
        Validation dataset (torch.utils.data.Dataset): The validation dataset
        Test dataset (torch.utils.data.Dataset): The test dataset
        num_classes (int): The number of classes in the dataset
        model (torch.nn.Module): The model to train
            Options:
                - Densenet
                - Resnet
                - EfficientNet
                - VisionTransformer
                - Custom CNN
        labels_map (dict): A dictionary mapping class indices to class names
        
        """
        self.train = train
        self.validation = validation
        self.test = test
        if len(self.validation) == 0:
            self.validation = self.test
        self.num_classes = num_classes
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.labels_map = labels_map

        if model == 'Densenet':
            self.model = densenet121(pretrained = False,num_classes=num_classes).to(self.device)
        elif model == 'Resnet':
            self.model = resnet50(pretrained = False,num_classes=num_classes).to(self.device)
        elif model == 'EfficientNet':
            self.model = efficientnet_b0(pretrained = False,num_classes=num_classes).to(self.device)
        elif model == 'VisionTransformer':
            self.model = vit_b_16(pretrained = False,num_classes=num_classes).to(self.device)
        elif model == 'Custom CNN':
            self.model = self.CustomCNN(num_classes=num_classes).to(self.device)
        else:
            raise ValueError("Invalid model name. Choose from ['Densenet', 'Resnet', 'EfficientNet', 'VisionTransformer', 'Custom CNN']")
        
    def CustomCNN(self, num_classes):
        # class CustomCNN(nn.Module):
        #     def __init__(self, num_classes):
        #         super(CustomCNN, self).__init__()
        #         self.conv1 = nn.Conv2d(3, 32, 3, 1)
        #         self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #         self.conv3 = nn.Conv2d(64, 128, 3, 1)
        #         self.fc1 = nn.Linear(128 * 10 * 10, 128)
        #         self.fc2 = nn.Linear(128, num_classes)

        #     def forward(self, x):
        #         x = F.relu(self.conv1(x))
        #         x = F.relu(self.conv2(x))
        #         x = F.relu(self.conv3(x))
        #         x = F.max_pool2d(x, 2)
        #         x = torch.flatten(x, 1)
        #         x = F.relu(self.fc1(x))
        #         x = self.fc2(x)
        #         return F.log_softmax(x, dim=1)

        # return CustomCNN(num_classes)
        raise NotImplementedError("Custom CNN model is in the making, and not ready for use yet.")
    
    def get_optimizer(self, optimizer_name='Adam', learning_rate=0.001, weight_decay=0):
        """
        Args:
            optimizer_name (str): Name of the optimizer to use.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
        """
        if optimizer_name == 'SGD':
            return optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name == 'Adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'RMSprop':
            return optim.RMSprop(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError("Invalid optimizer name. Choose from ['SGD', 'Adam', 'AdamW', 'RMSprop']")
        
    def train_model(self, epochs, learning_rate=0.001,optimizer="Adam", weight_decay=0):
        """
        Trains the model on the training dataset.
        Args:
            epochs (int): Number of epochs to train the model.
            learning_rate (float): Learning rate for the optimizer.
            optimizer (str): Name of the optimizer to use.
            weight_decay (float): Weight decay for the optimizer.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = self.get_optimizer(optimizer_name=optimizer, learning_rate=learning_rate, weight_decay=weight_decay)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss, epoch_acc = 0, 0

            for batch in tqdm(self.train):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                epoch_acc += torch.sum(preds == labels.data).item()

            train_loss = epoch_loss / len(self.train.dataset)
            train_acc = epoch_acc / len(self.train.dataset)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation phase using evaluation func
            val_results = self.evaluate(dataset=self.validation)
            val_loss = val_results['loss']
            val_acc = val_results['accuracy']
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    
    def plot_history(self):
        """
        Plots the training history of the model.
        """
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], label='Train Accuracy')
        plt.plot(epochs, self.history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
    
    def evaluate(self, dataset=None):
        """
        Evaluates the model's performance on a given dataset.

        Args:
            dataset (torch.utils.data.DataLoader): Dataset to evaluate on (default is test set).

        Returns:
            dict: A dictionary containing accuracy and loss.
        """
        if dataset is None:
            dataset = self.test  # Default to the test set

        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()  

        with torch.no_grad():  # Disable gradient calculation for evaluation
            for inputs, labels in tqdm(dataset, desc="Evaluating", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)  
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        average_loss = total_loss / len(dataset)

        return {"accuracy": accuracy, "loss": average_loss}
    
    def predict(self, inputs):
        """
        Makes predictions using the trained model for given inputs.

        Args:
            inputs (torch.Tensor): Input images (batch or single image).

        Returns:
            torch.Tensor: Predicted class indices.
        """
        self.model.eval()  
        
        with torch.no_grad():  # Disable gradient calculation
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1) 
        return predicted.cpu() #ensure its moved to cpu just in case its on gpu

    def save_model(self, path):
        """
        Saves the model to file

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self,path):
        """
        Load model from saved file

        Args:
            path (str): Path to load the model.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def plot_img(self, indexes =(0,4) ,prediction = False):
        """
        Args:
            indexes (tuple): Tuple of two integers representing the range of images to plot.
            prediction (bool): If True, the model will make predictions on the images, and plot the predicted labels.
        """
        images, labels = next(iter(self.test))
        images, labels = images[indexes[0]:indexes[1]], labels[indexes[0]:indexes[1]]
        num_images = indexes[1] - indexes[0]
        if prediction:
            self.predicted_labels = self.predict(images) 
            print(self.predicted_labels)
        
        self.plot_batch(images, labels, self.labels_map, num_images)
        
    
    def plot_batch(self,images, labels, labels_map, num_images):
        """
        Plots a grid of images with their corresponding labels.

        Args:
            images (Tensor): Batch of images (Shape: [batch_size, channels, height, width]).
            labels (Tensor): Corresponding labels for the images.
            labels_map (dict): Mapping from label indices to class names.
            num_images (int): Number of images to display (default is 9).
            prediction_label (Tensor): Predicted labels for the images.
        """
        # Limit the number of images to the square root for a clean grid
        grid_size = int(np.sqrt(num_images))
        num_images = grid_size ** 2

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        for i, ax in enumerate(axes.flatten()):
            if i >= num_images:
                break
            img = images[i].permute(1, 2, 0).numpy()
            label = labels[i].item()
            ax.imshow(img)
            label_title = f"{labels_map[label]}"
            
            if hasattr(self, 'predicted_labels'):  # Check if predictions exist
                labelpred = self.predicted_labels[i].item()  # Get prediction from self.predicted_labels
                label_title += f"\nPrediction: {labels_map[labelpred]}"
            
            ax.set_title(label_title)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    # Parameters
    num_classes = 4
    epochs = 2
    learning_rate = 0.001
    batch_size = 32
    base_path = "Data/archive"
    image_size = (32, 32)

    # Load and preprocess the MRI data
    print("Loading data...")#Data/archive/Testing

    total_dataset, labels_map = load_mri_data(base_path, batch_size=batch_size, image_size=image_size)

    # Split into training and testing datasets
    train_set, val_set ,test_set = train_test_split(total_dataset, test_fraction=0.2, batch_size=batch_size)

    # Initialize and train models
    models = ["Densenet", "Resnet", "EfficientNet", "VisionTransformer"]
    histories = {}

    for model_name in models:
        print(f"\nTraining {model_name} model...\n")
        print("initializing model...")
        cnn_model = CNN(train_set, val_set, test_set, num_classes, model_name,labels_map)
        print("Training model...")
        cnn_model.train_model(epochs=epochs, learning_rate=learning_rate, optimizer="Adam", weight_decay=0)
        histories[model_name] = cnn_model.history

        # Plot training history
        print(f"Plotting training history for {model_name}...")
        cnn_model.plot_history()

        # Plot sample images with predictions
        print(f"Plotting predictions for {model_name}...")
        cnn_model.plot_img(indexes=(0, 4), prediction=True)

    print("\nAll models trained and evaluated!")