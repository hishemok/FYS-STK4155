import autograd.numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
np.random.seed(0)
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import read_data
from nn.neural_network import NeuralNetwork,ReLU, ReLU_derivative, leaky_ReLU, leaky_ReLU_derivative, sigmoid, sigmoid_derivative, MSE, MSE_derivative, R2, softmax



def prep():
    data = read_data()
    #last row is nan and first row is the id
    data = data[:,1:-1]
    #in the dataset 'M' and 'B' are used to represent the classes, we will change them to 1 and 0
    data[data == 'M'] = 1
    data[data == 'B'] = 0
    return data


def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)
    print(one_hot_predictions.shape)
    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)

def cross_entropy(predict, target):
    predict = np.clip(predict, 1e-15, 1 - 1e-15)
    return np.sum(-target * np.log(predict))

def binary_cross_entropy(predict, target):
    # Clip values to avoid log(0)
    predict = np.clip(predict, 1e-10, 1 - 1e-10)
    return -np.mean(target * np.log(predict) + (1 - target) * np.log(1 - predict))

if __name__ == "__main__":

    print("Running neural network on breast cancer dataset")
    data = prep()


    X = data[:, 1:]  # Features
    y = data[:, 0].astype(int)   # Labels (0 for benign, 1 for malignant)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    hidden_sizes = [[10, 5], [5, 2], [5,4], [10,2] ,[2]]
    
    for hidden_size in hidden_sizes:
        nn = NeuralNetwork(input_size=30, hidden_sizes=hidden_size, output_size=1,
                           hidden_layer_activation_function=[sigmoid,softmax] , activation_der=[sigmoid_derivative,lambda x:1],
                           output_layer_activation_functions=sigmoid, cost_function=MSE, cost_der=MSE_derivative)

        nn.train(X_train, y_train.reshape(-1, 1), n_epochs=200, learning_rate=0.001, batch_size=32)

        # Prediction
        preds = nn.forward(X_test)
        preds_binary = (preds > 0.5).astype(int).flatten()

        # Accuracy calculation
        acc = accuracy_score(y_test, preds_binary)
        print(f'Hidden sizes: {hidden_size}')
        print(f'Accuracy: {acc * 100:.4f}%')
        nn.reset_weights()



