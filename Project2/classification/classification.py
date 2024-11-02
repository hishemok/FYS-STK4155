import autograd.numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
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

def binary_cross_entropy_derivative(predict, target):
    return (predict - target)

if __name__ == "__main__":

    print("Running neural network on breast cancer dataset")
    data = prep()


    X = data[:, 1:]  # Features
    y = data[:, 0].astype(int)   # Labels (0 for benign, 1 for malignant)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    hidden_size = [10,5]
    learning_rate = 0.1
    batch_size = 64
    cost_functions = [MSE,binary_cross_entropy]
    cost_derivs = [MSE_derivative,binary_cross_entropy_derivative]
    hidden_functions = [[ReLU,ReLU],[sigmoid,sigmoid],[softmax,softmax],[leaky_ReLU,leaky_ReLU]
                        ,[ReLU,leaky_ReLU],[ReLU,sigmoid],[ReLU,softmax],
                        [leaky_ReLU,softmax],[leaky_ReLU,sigmoid],[sigmoid,softmax]
                        ,[leaky_ReLU,ReLU],[sigmoid,ReLU],[softmax,ReLU],
                        [softmax,leaky_ReLU],[sigmoid,leaky_ReLU],[softmax,sigmoid]]
    #hidden derivatives match the activation functions
    hidden_derivatives = [[ReLU_derivative,ReLU_derivative],[sigmoid_derivative,sigmoid_derivative],[lambda x:1,lambda x:1],[leaky_ReLU_derivative,leaky_ReLU_derivative], 
                            [ReLU_derivative,leaky_ReLU_derivative],[ReLU_derivative,sigmoid_derivative],[ReLU_derivative,lambda x:1],
                            [leaky_ReLU_derivative,lambda x:1],[leaky_ReLU_derivative,sigmoid_derivative],[sigmoid_derivative,lambda x:1],
                            [leaky_ReLU_derivative,ReLU_derivative],[sigmoid_derivative,ReLU_derivative],[lambda x:1,ReLU_derivative],
                            [lambda x:1,leaky_ReLU_derivative],[sigmoid_derivative,leaky_ReLU_derivative],[lambda x:1,sigmoid_derivative]]
    output_functions = [sigmoid,softmax,ReLU,leaky_ReLU]

    best_params = {'hidden_size':[],'learning_rate':[],'batch_size':[],'hidden_funcs':[],'hidden_derivs':[],'output_func':[],'cost_func':[],'cost_derivative': [],'accuracy':0}
    results = []
    for cost_func in cost_functions:
        for cost_der in cost_derivs:
            for i in range(len(hidden_functions)):
                for out in output_functions:
                    nn = NeuralNetwork(input_size=30, hidden_sizes=hidden_size, output_size=1,
                            hidden_layer_activation_function=hidden_functions[i], activation_der=hidden_derivatives[i],
                            output_layer_activation_functions=out, cost_function=cost_func, cost_der=cost_der)
                    
                    nn.train(X_train, y_train.reshape(-1, 1), n_epochs=200, learning_rate=learning_rate, batch_size=batch_size)

                    # Prediction
                    preds = nn.forward(X_test)
                    preds_binary = (preds > 0.5).astype(int).flatten()

                    # Accuracy calculation
                    acc = accuracy_score(y_test, preds_binary)

                    if acc > best_params['accuracy']:
                        best_params['hidden_size'] = hidden_size
                        best_params['learning_rate'] = learning_rate
                        best_params['batch_size'] = batch_size
                        HF = [func.__name__ for func in nn.hidden_layer_activation_function]
                        HFder = [func.__name__ for func in nn.activation_derivative]
                        best_params['hidden_funcs'] = HF
                        best_params['hidden_derivs'] = HFder
                        best_params['output_func'] = out.__name__
                        best_params['cost_func'] = cost_func.__name__
                        best_params['cost_derivative'] = cost_der.__name__
                        best_params['accuracy'] = acc

                    #Write results to file
                    result = {

                        'Hidden Activation Functions': [func.__name__ for func in hidden_functions[i]],
                        'Hidden Derivatives': [func.__name__ for func in hidden_derivatives[i]],
                        'Output Activation Function': out.__name__,
                        'Cost Function': cost_func.__name__,
                        'Cost Derivative': cost_der.__name__,
                        'Accuracy (%)': acc * 100
                    }

                    # Add dictionary to results list
                    results.append(result)

    # Convert results list to a pandas DataFrame
    df_results = pd.DataFrame(results)

    # Save DataFrame to a CSV file
    df_results.to_csv('Best_functions.csv', index=False)

    # Display the best parameters
    print(f'Best Parameters:\n{df_results.loc[df_results["Accuracy (%)"].idxmax()]}')



 