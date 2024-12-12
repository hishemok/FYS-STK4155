import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.models import densenet121, resnet50, efficientnet_b0, vit_b_16, mobilenet_v2, mobilenet_v3_large
from Data.load_data import load_mri_data, train_test_split 
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
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
                - MobileNetV2
                - MobileNetV3

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
        
        elif model == 'MobileNetV2':
            self.model = mobilenet_v2(weights = False,num_classes=num_classes).to(self.device)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

            self.model_name = 'MobileNetV2'
        
        elif model == 'MobileNetV3':
            self.model = mobilenet_v3_large(weights = False,num_classes=num_classes).to(self.device)
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)

            self.model_name = 'MobileNetV3'

        else:
            raise ValueError("Invalid model name. Choose from ['Densenet', 'Resnet', 'EfficientNet', 'VisionTransformer', 'MobileNetV2', 'MobileNetV3']")
        

    
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
        self.optimizer_name = kwargs.get('optimizer', 'Adam')

        self.epochs = epochs
        self.lr = learning_rate

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



    def predict(self, dataset):
        """
        Predicts the labels for a given dataset.
        Params:
            dataset (torch.utils.data.Dataset): The dataset to predict on.
        Returns:
            predictions (Tensor): Predicted labels.
            probabilities (Tensor): Class probabilities.
            lbls (Tensor): True labels
        """
        self.model.eval()
        predictions = []
        probabilities = []
        lbls = []

        with torch.no_grad():
            for batch,labels in dataset:
                inputs = batch.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                probabilities.extend(torch.softmax(outputs, dim=1))
                lbls.extend(labels)
                predictions.extend(preds.cpu().numpy())
        
        return predictions, probabilities, lbls




    def evaluate(self, dataset):
        """
        Evaluates the model's performance on a given dataset.

        Params:
            dataset (torch.utils.data.DataLoader): Dataset to evaluate on (default is test set).

        Returns:
            dict: A dictionary containing accuracy and loss.
        """
        dataset = dataset if dataset is not None else self.test

        self.model.eval()  # Set the model to evaluation mode


        predictions, probabilities, lbls = self.predict(dataset)

        predictions = torch.tensor(predictions)
        lbls = torch.tensor(lbls)

    
        correct = (predictions == lbls).sum().item()
        total_loss = (predictions != lbls).sum().item()

        total = len(lbls)

        accuracy = correct / total

        loss = total_loss / len(dataset.dataset)

        return {'accuracy': accuracy, 'loss': loss}

    def r2_mse(self, dataset):
        """
        Calculates the R² and MSE for a given dataset.
        Params:
            dataset (torch.utils.data.Dataset): The dataset to evaluate.
        Returns:
            r2 (float): The R² value.
            mse (float): The mean squared error.
        """
        predictions, probabilities, lbls = self.predict(dataset)

        predictions = torch.tensor(predictions)
        lbls = torch.tensor(lbls)

        r2 = r2_score(lbls, predictions)
        mse = mean_squared_error(lbls, predictions)

        return r2, mse
    

    def predict_xgb(self,save = False, path = "Data/Results"):
        """
        Predicts the labels using XGBoost.
        """


        # Get the predictions and probabilities
        predictions_train, probabilities_train, lbls_train = self.predict(self.train)
        predictions_test, probabilities_test, lbls_test = self.predict(self.test)


        # Convert the data to numpy arrays
        predictions_train = np.array(predictions_train)
        probabilities_train = np.array(probabilities_train)
        lbls_train = np.array(lbls_train)

        predictions_test = np.array(predictions_test)
        probabilities_test = np.array(probabilities_test)
        lbls_test = np.array(lbls_test)

        # Train the XGBoost model
        xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=self.num_classes)
        xgb_model.fit(probabilities_train, lbls_train)

        # Get the predictions using the XGBoost model
        xgb_predictions = xgb_model.predict(probabilities_test)
        


        xgb_accuracy = np.mean(xgb_predictions == lbls_test)

        r2_xgb = r2_score(lbls_test, xgb_predictions)
        mse_xgb = mean_squared_error(lbls_test, xgb_predictions)

        print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
        print(f"XGBoost R²: {r2_xgb:.4f}")
        print(f"XGBoost MSE: {mse_xgb:.4f}")

        # save accuracy, r2 and mse
        if save:
            path1 = path + f"/XGBoost_metrics_{self.model_name}_eps_{self.epochs}_lr_{self.lr}_opt_{self.optimizer_name}.txt"
            with open(path1, 'w') as f:
                f.write(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
                f.write(f"XGBoost R²: {r2_xgb:.4f}")
                f.write(f"XGBoost MSE: {mse_xgb:.4f}")


        report = classification_report(lbls_test, xgb_predictions, target_names=self.labels_map.values())
        print(report)

        if save:
            pathc = path + f"/classification_report_{self.model_name}_eps_{self.epochs}_lr_{self.lr}_opt_{self.optimizer_name}.txt"
            with open(pathc, 'w') as f:
                f.write(report)

        cm = confusion_matrix(lbls_test, xgb_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=self.labels_map.values(), yticklabels=self.labels_map.values())
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f"Confusion Matrix for XGBoost")
        if save:
            path = path + f"/confusion_matrix_XGBoost_{self.model_name}_eps_{self.epochs}_lr_{self.lr}_opt_{self.optimizer_name}.png"
            plt.savefig(path)
        
        plt.show(block=False)



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
            path = path + f"/training_history_{self.model_name}_eps_{self.epochs}_lr_{self.lr}_opt_{self.optimizer_name}.png"
            plt.savefig(path)
        plt.show(block=False)

    
    def plot_confusion_matrix(self, dataset, save=False, path="Data/Results"):
        """
        Plots the confusion matrix for the model.
        Params:
            dataset (torch.utils.data.Dataset): The dataset to evaluate.
            save (bool): If True, saves the plot (default is False).
            path (str): The path to save the plot (default is 'Data/Results').
        """
        predictions, probabilities, lbls = self.predict(dataset)

        predictions = torch.tensor(predictions)
        lbls = torch.tensor(lbls)

        cm = confusion_matrix(lbls, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=self.labels_map.values(), yticklabels=self.labels_map.values())
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f"Confusion Matrix for {self.model_name}")
        if save:
            path = path + f"/confusion_matrix_{self.model_name}_eps_{self.epochs}_lr_{self.lr}_opt_{self.optimizer_name}.png"
            plt.savefig(path)
        plt.show(block=False)

    
    def classification_report(self, dataset, save=False, path="Data/Results"):
        """
        Prints the classification report for the model.
        Params:
            dataset (torch.utils.data.Dataset): The dataset to evaluate.
            save (bool): If True, saves the report to a file (default is False).
            path (str): The path to save the report (default is 'Data/Results').
        """
        predictions, probabilities, lbls = self.predict(dataset)

        predictions = torch.tensor(predictions)
        lbls = torch.tensor(lbls)

        report = classification_report(lbls, predictions, target_names=self.labels_map.values())
        print(report)

        if save:
            path = path + f"/classification_report_{self.model_name}_eps_{self.epochs}_lr_{self.lr}_opt_{self.optimizer_name}.txt"
            with open(path, 'w') as f:
                f.write(report)

    
    def save_model(self, path="Data/Results"):
        """
        Saves the model to a file.
        Params:
            path (str): The path to save the model (default is 'Data/Results').
        """
        path = path + f"/{self.model_name}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Loads a model from a file.
        Params:
            path (str): The path to load the model from.
        """
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")





if __name__ == "__main__":
    # Parameters
    num_classes = 4
    epochs = 2
    learning_rate = 0.001
    batch_size = 32
    base_path = "Data/archive"
    image_size = (128, 128)



 
    models = ["MobileNetV3"]
    epochs = 20

    for model in models:

        print("Loading data...")
        total_dataset, labels_map = load_mri_data(base_path, batch_size=batch_size, image_size=image_size)
        train_set, val_set, test_set = train_test_split(total_dataset, test_fraction=0.2, batch_size=batch_size)
        
        cnn_model = CNN(
            train=train_set,
            validation=val_set,
            test=test_set,
            num_classes=num_classes,
            model=model,
            labels_map=labels_map,
            image_size=image_size[0]
        )
        print(f"Training {model}...")
        cnn_model.train_model(epochs=epochs, learning_rate=learning_rate, optimizer="Adam")
        cnn_model.plot_history(save=True)
        cnn_model.classification_report(test_set, save=True)
        cnn_model.plot_confusion_matrix(test_set, save=True)
        cnn_model.predict_xgb(save=True, path="Data/Results/xgboost")

# # Load and preprocess the MRI data
    # print("Loading data...")#Data/archive/Testing

    # total_dataset, labels_map = load_mri_data(base_path, batch_size=batch_size, image_size=image_size)

    # # Split into training and testing datasets
    # train_set, val_set ,test_set = train_test_split(total_dataset, test_fraction=0.2, batch_size=batch_size)

    # model = "Densenet"


    

    # #initate the model
    # cnn_model = CNN(
    #     train=train_set,
    #     validation=val_set,
    #     test=test_set,
    #     num_classes=num_classes,
    #     model=model,
    #     labels_map=labels_map,
    #     image_size=image_size
    # )

    # # Train the model
    # print(f"Training {model}...")
    # cnn_model.train_model(epochs=epochs,learning_rate=learning_rate,optimizer="Adam")

    # #predict on test
    # predictions, probabilities, lbls = cnn_model.predict(test_set)


    # #plot history
    # cnn_model.plot_history(save = True)

    # #try xgboost
    # cnn_model.predict_xgb(save = True, path= "Data/Results/xgboost")

