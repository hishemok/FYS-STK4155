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
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import xgboost as xgb


class CNN:
    def __init__(self, train, validation, test, num_classes,model,labels_map,**kwargs):
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
        labels_map (dict): A dictionary mapping class indices to class names
        kwargs:
            - image_size (int): The size of the input images (default is 32)
        
        """
        self.train = train
        self.validation = validation if len(validation) > 0 else test
        self.test = test
        self.num_classes = num_classes
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'r2': [], 'mse': []}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.labels_map = labels_map


        if model == 'Densenet':
            self.model = densenet121(weights = False,num_classes=num_classes).to(self.device)
            #Customize classifier for the number of classes in the dataset
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

            self.model_name = 'Densenet'
        elif model == 'Resnet':
            self.model = resnet50(weights = False,num_classes=num_classes).to(self.device)
            #Customize classifier for the number of classes in the dataset, resnet50 has fc layer
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

            self.model_name = 'Resnet'
        elif model == 'EfficientNet':
            self.model = efficientnet_b0(weights = False,num_classes=num_classes).to(self.device)
            #Customize classifier for the number of classes in the dataset
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2,inplace=True),
                nn.Linear(in_features, num_classes)
            )

            self.model_name = 'EfficientNet'
        elif model == 'VisionTransformer':
            self.model = vit_b_16(weights = False,num_classes=num_classes, image_size=kwargs["image_size"]).to(self.device)
            self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)

            self.model_name = 'VisionTransformer'

        else:
            raise ValueError("Invalid model name. Choose from ['Densenet', 'Resnet', 'EfficientNet', 'VisionTransformer']")
        

    
    def get_optimizer(self, optimizer_name='Adam', learning_rate=0.001, weight_decay=0):
        """
        Params:
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
        
    def train_model(self, epochs,**kwargs):#learning_rate=0.001,optimizer="Adam", weight_decay=0
        """
        Trains the model on the training dataset.
        Params:
            epochs (int): Number of epochs to train the model.
            kwargs: 
                -learning_rate (float): Learning rate for the optimizer.
                -optimizer (str): Name of the optimizer to use.
                -weight_decay (float): Weight decay for the optimizer.
                -return_mse_r2 (bool): If True, returns the R² and MSE values for the validation set, across epochs.
                -use_scheduler (bool): If True, uses a learning rate scheduler.
                    -scheduler_mode (str): Mode for the scheduler (default is 'min').
                    -factor (float): Factor by which to reduce the learning rate (default is 0.1).
                    -patience (int): Number of epochs with no improvement after which learning rate will be reduced (default is 5).
                    -verbose (bool): If True, prints the learning rate updates (default is True).

        """
        learning_rate = kwargs.get('learning_rate', 0.001)
        optimizer = kwargs.get('optimizer', 'Adam')
        weight_decay = kwargs.get('weight_decay', 0)
        use_scheduler = kwargs.get('use_scheduler', False)


        criterion = nn.CrossEntropyLoss()
        optimizer = self.get_optimizer(optimizer_name=optimizer, learning_rate=learning_rate, weight_decay=weight_decay)


        if use_scheduler:
            scheduler_mode = kwargs.get('scheduler_mode', 'min')
            factor = kwargs.get('factor', 0.1)
            patience = kwargs.get('patience', 5)
            verbose = kwargs.get('verbose', True)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=factor, patience=patience, verbose=verbose)

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
            
            if use_scheduler:
                scheduler.step(metrics=loss)

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
            r2, mse = self.r2_mse(self.validation)
            self.history['r2'].append(r2)
            self.history['mse'].append(mse)

            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
            print(f"  Val R²: {r2:.4f}, Val MSE: {mse:.4f}")

 
        
    def plot_history(self,save = False,path = "Data/Results"):
        """
        Plots the training history of the model.
        """
        epochs = range(1, len(self.history['train_loss']) + 1)
        # plt.figure(figsize=(12, 5))

        fig, axs = plt.subplots(2,2, figsize=(12, 12))

        axs[0, 0].plot(epochs, self.history['train_loss'], label='Train Loss')
        axs[0, 0].plot(epochs, self.history['val_loss'], label='Validation Loss')
        # axs[0, 0].set_xlabel('Epochs')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()

        axs[0, 1].plot(epochs, self.history['train_acc'], label='Train Accuracy')
        axs[0, 1].plot(epochs, self.history['val_acc'], label='Validation Accuracy')
        # axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('Accuracy')
        axs[0, 1].legend()


        axs[1, 0].plot(epochs, self.history['r2'], label='Validation R²')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('R²')
        axs[1, 0].legend()

        axs[1, 1].plot(epochs, self.history['mse'], label='Validation MSE')
        axs[1, 1].set_xlabel('Epochs')
        axs[1, 1].set_ylabel('MSE')
        axs[1, 1].legend()

        plt.title(f"Training History for {self.model_name}")

        if save:
            path = path + f"/training_history_{self.model_name}.png"
            plt.savefig(path)
        plt.show(block=False)
    
    def evaluate(self, dataset=None):
        """
        Evaluates the model's performance on a given dataset.

        Params:
            dataset (torch.utils.data.DataLoader): Dataset to evaluate on (default is test set).

        Returns:
            dict: A dictionary containing accuracy and loss.
        """
        dataset = dataset if dataset is not None else self.test

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

        Params:
            inputs (torch.Tensor): Input images (batch or single image).

        Returns:
            torch.Tensor: Predicted class indices.
            torch.Tensor: Predicted class probabilities.
        """
        self.model.eval()  
        
        with torch.no_grad():  # Disable gradient calculation
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1) 
        return predicted.cpu(),probabilities.cpu() #ensure its moved to cpu just in case its on gpu


    


    def predict_xgb(self,**kwargs):
        """
        Predict using XGBoost model.
        Params:

        Returns:
            torch.Tensor: Predicted class indices.
            torch.Tensor: Predicted class probabilities.

        """
        # Load the XGBoost model
        cnn_model = self.model
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        max_depth = kwargs.get('max_depth', 6)
        n_estimators = kwargs.get('n_estimators', 100)
        learning_rate = kwargs.get('learning_rate', 0.1)
        save = kwargs.get('save', False)
        path = kwargs.get('path', "Data/Results/xgboost")

        def extract_features(dataset, model, device):
            model.eval()
            features, labels = [], []
            with torch.no_grad():
                for inputs, lbls in tqdm(dataset, desc="Extracting features"):
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    features.append(outputs.cpu().numpy())
                    labels.append(lbls.numpy())
            return np.concatenate(features), np.concatenate(labels)
        

        
        X_train, y_train = extract_features(self.train, cnn_model, device)
        X_val, y_val = extract_features(self.validation, cnn_model, device)
        X_test, y_test = extract_features(self.test, cnn_model, device)



        xgb_model = xgb.XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            objective='multi:softmax',
            num_class=self.num_classes,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

        xgb_preds = xgb_model.predict(X_test)


        r2, mse = self.r2_mse(y_test, xgb_preds)
        
        cm = confusion_matrix(y_test, xgb_preds,labels = self.labels_map)
        class_names = list(self.labels_map.values())
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=class_names,yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix on testset from the XGBoost model with model {self.model_name}')
        plt.show(block=False)

        class_report = classification_report(y_test, xgb_preds, target_names=self.labels_map.values())

        if save:
            paths = path + f"/xgb_{self.model_name}.png"
            plt.savefig(paths)
            with open(f"{path}/xgb_classification_report_mod_{self.model_name}.txt", 'w') as f:
                f.write(class_report)

            np.save(f"{path}/xgb_r2.npy", r2)
            np.save(f"{path}/xgb_mse.npy", mse)


        
        return xgb_preds

    def save_model(self, path):
        """
        Saves the model to file

        Params:
            path (str): Path to save the model.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self,path):
        """
        Load model from saved file

        Params:
            path (str): Path to load the model.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def plot_img(self, indexes =(0,4) ,prediction = False,save = False, path = "Data/Results"):
        """
        Params:
            indexes (tuple): Tuple of two integers representing the range of images to plot.
            prediction (bool): If True, the model will make predictions on the images, and plot the predicted labels.
        """
        images, labels = next(iter(self.test))
        images, labels = images[indexes[0]:indexes[1]], labels[indexes[0]:indexes[1]]
        num_images = indexes[1] - indexes[0]
        if prediction:
            self.predicted_labels,self.probabilities = self.predict(images) 
            print(self.predicted_labels,self.probabilities)
        
        self.plot_batch(images, labels, self.labels_map, num_images,save=save,path=path)
        
    
    def plot_batch(self,images, labels, labels_map, num_images,save = False, path = "Data/Results"):
        """
        Plots a grid of images with their corresponding labels.

        Params:
            images (Tensor): Batch of images (Shape: [batch_size, channels, height, width]).
            labels (Tensor): Corresponding labels for the images.
            labels_map (dict): Mapping from label indices to class names.
            num_images (int): Number of images to display (default is 9).
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
                labelprob = self.probabilities[i].numpy()
                label_title += f"\nPrediction: {labels_map[labelpred]} {labelprob[labelpred]:.2f}%"
            
            ax.set_title(label_title)

        plt.tight_layout()
        if hasattr(self, 'predicted_labels'):
            plt.title(f"Sample Images from the {self.model_name} Model")
        else:
            plt.title("Sample Images")

        if save:
            path += f"/sample_images_{self.model_name}.png"
            plt.savefig(path)

        plt.show(block=False)
    
    def plot_confusion_matrix(self,dataset,save = False,path = "Data/Results"):
        """
        Plots the confusion matrix for the model on a given dataset.

        Params:
            dataset (torch.utils.data.DataLoader): Dataset to evaluate on (default is test set).
            Path (str): Path to save the confusion matrix plot.
        """
        y_true = [] 
        y_pred = []
        for inputs, labels in dataset:
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        cm = confusion_matrix(y_true, y_pred)
        class_names = list(self.labels_map.values())
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=class_names,yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix on testset from the {self.model_name} model')
        if save:
            path = path + f"/confusion_matrix_{self.model_name}.png"
            plt.savefig(path) 
        plt.show(block=False)
    
    def classification_report(self,dataset,save = False,path = "Data/Results"):
        """
        Prints the classification report for the model on a given dataset.

        Params:
            dataset (torch.utils.data.DataLoader): Dataset to evaluate on (default is test set).
        """
        y_true = [] 
        y_pred = []
        for inputs, labels in dataset:
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        print(classification_report(y_true, y_pred, target_names=self.labels_map.values()))

        if save:
            path = path + f"/classification_report_{self.model_name}.txt"
            with open(path, 'w') as f:
                f.write(classification_report(y_true, y_pred, target_names=self.labels_map.values()))
    
    def r2_mse(self, dataset):
        """
        Calculates R² and MSE for the model's predictions on a dataset.

        Params:
            dataset (torch.utils.data.DataLoader): Dataset to evaluate on (default is test set).

        Returns:
            float: R² value
            float: Mean Squared Error (MSE)
            
        """
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for inputs, labels in dataset:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mse = np.mean((y_true - y_pred) ** 2)
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

        return r2, mse

    def get_metrics_and_plots(self,**kwargs):
        """
        Get metrics and plots for the trained model.
        Params:
            kwargs:
                -saving (bool): If True, save the plots to file (default is False).
                -path (str): Path to save the plots (default is "Data/Results").
                - confusion_matrix (bool): If True, plot the confusion matrix (default is True).
                - classification_report (bool): If True, print the classification report (default is True).
                - plot_history (bool): If True, plot the training history (default is True).
                - r2_mse (bool): If True, calculate R² and MSE (default is True).
                - plot_img (bool): If True, plot sample images with predictions (default is False).
                    - indexes (tuple): Tuple of two integers representing the range of images to plot (default is (0, 4)).
                    - prediction (bool): If True, the model will make predictions on the images, and plot the predicted labels (default is True).
        """
        saving = kwargs.get('saving', False)
        path = kwargs.get('path', "Data/Results")
        confusion_matrix = kwargs.get('confusion_matrix', True)
        classification_report = kwargs.get('classification_report', True)
        plot_history = kwargs.get('plot_history', True) 
        plot_img = kwargs.get('plot_img', False)
        indexes = kwargs.get('indexes', (0, 4))
        prediction = kwargs.get('prediction', True)    


        if confusion_matrix:
            print("Plotting confusion matrix...")
            self.plot_confusion_matrix(self.test,save = saving,path = path)
        
        if classification_report:
            print("Printing classification report...")
            self.classification_report(self.test,save = saving,path = path)

        if plot_history:
            print("Plotting training history...")
            self.plot_history(save=saving,path=path)
        
        if plot_img:
            print("Plotting sample images...")
            self.plot_img(indexes=indexes, prediction=prediction,save=saving,path=path)


if __name__ == "__main__":
    # Parameters
    num_classes = 4
    epochs = 2
    learning_rate = 0.001
    batch_size = 64
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
        
        # Initialize the CNN model
        cnn_model = CNN(
            train=train_set,
            validation=val_set,
            test=test_set,
            num_classes=num_classes,
            model=model_name,
            labels_map=labels_map,
            image_size=image_size  # Only applicable for VisionTransformer
        )

        # Train the model
        print(f"Training {model_name}...")
        cnn_model.train_model(
            epochs=epochs,
            learning_rate=learning_rate,
            optimizer="Adam"
        )
    

        # Save the model# Save training history
        histories[model_name] = cnn_model.history

        # Evaluate the model and plot metrics
        print(f"Evaluating {model_name}...")
        cnn_model.get_metrics_and_plots(confusion_matrix=True, classification_report=True, plot_history=True, plot_img=True, indexes=(0, 4), prediction=True)

        print(f"Predicting using XGBoost for {model_name}...")
        cnn_model.predict_xgb()

    print("\nAll models trained and evaluated!")