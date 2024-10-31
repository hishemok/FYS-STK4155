import autograd.numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import accuracy_score
np.random.seed(0)
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.frankefunction import franke_function
from nn.neural_network import NeuralNetwork, ReLU, ReLU_derivative, leaky_ReLU, leaky_ReLU_derivative, sigmoid, sigmoid_derivative, MSE, MSE_derivative, R2


def prep():
    x = np.linspace(0,1,100)#np.random.rand(10000, 1)#
    y = np.linspace(0,1,100)#
    x,y = np.meshgrid(x,y)
    x = x.flatten().reshape(-1,1)
    y = y.flatten().reshape(-1,1)

    z = franke_function(x, y)

    # Normalize inputs
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(x)
    Y_scaled = scaler_y.fit_transform(y)

    # Prepare inputs for the neural network
    inputs = np.hstack((X_scaled, Y_scaled))
    z = (z - np.mean(z)) / np.std(z)  # Standardize target
    inputs = np.hstack((x,y))
    print(inputs.shape)
    return inputs, z, x, y


def sigmoid_test():
    inputs,z,_,_ = prep()

    nn = NeuralNetwork(input_size=2, hidden_sizes=[50, 25], output_size=1)
    nn.train(inputs, z.reshape(-1, 1), n_epochs=2000, learning_rate=0.001, batch_size=32) 

    preds = nn.forward(inputs)  
    mse = MSE(preds, z)
    r2 = R2(preds, z)

    print(f"Sigmoid Activation Function")
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')

def relu_test():
    inputs,z,_,_ = prep()

    nn = NeuralNetwork(input_size=2, hidden_sizes=[50, 25], output_size=1, hidden_layer_activation_function=ReLU, activation_der=ReLU_derivative)
    nn.train(inputs, z.reshape(-1, 1), n_epochs=2000, learning_rate=0.001, batch_size=32)

    preds = nn.forward(inputs)
    mse = MSE(preds, z)
    r2 = R2(preds, z)

    print(f"ReLU Activation Function")
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')

def leaky_relu_test():
    inputs,z,_,_ = prep()

    nn = NeuralNetwork(input_size=2, hidden_sizes=[50, 25], output_size=1, hidden_layer_activation_function=leaky_ReLU, activation_der=leaky_ReLU_derivative)
    nn.train(inputs, z.reshape(-1, 1), n_epochs=2000, learning_rate=0.001, batch_size=32)


    preds = nn.forward(inputs)
    mse = MSE(preds, z)
    r2 = R2(preds, z)

    print(f"Leaky ReLU Activation Function")
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')


def all_off_them():
    inputs,z,_,_ = prep()

    nn = NeuralNetwork(input_size=2, hidden_sizes=[50, 25], output_size=1)
    nn.train(inputs, z.reshape(-1, 1), n_epochs=2000, learning_rate=0.001, batch_size=32) 

    preds = nn.forward(inputs)  
    mseSIG = MSE(preds, z)
    r2SIG = R2(preds, z)

    print(f"Sigmoid Activation Function")
    print(f'Mean Squared Error: {mseSIG}')
    print(f'R² Score: {r2SIG}')

    nn = NeuralNetwork(input_size=2, hidden_sizes=[50, 25], output_size=1, hidden_layer_activation_function=ReLU, activation_der=ReLU_derivative)
    nn.train(inputs, z.reshape(-1, 1), n_epochs=2000, learning_rate=0.001, batch_size=32)

    preds = nn.forward(inputs)
    mseRE = MSE(preds, z)
    r2RE = R2(preds, z)

    print(f"ReLU Activation Function")
    print(f'Mean Squared Error: {mseRE}')
    print(f'R² Score: {r2RE}')

    nn = NeuralNetwork(input_size=2, hidden_sizes=[50, 25], output_size=1, hidden_layer_activation_function=leaky_ReLU, activation_der=leaky_ReLU_derivative)
    nn.train(inputs, z.reshape(-1, 1), n_epochs=2000, learning_rate=0.001, batch_size=32)


    preds = nn.forward(inputs)
    mseLRE = MSE(preds, z)
    r2LRE = R2(preds, z)

    print(f"Leaky ReLU Activation Function")
    print(f'Mean Squared Error: {mseLRE}')
    print(f'R² Score: {r2LRE}')

    #write to file
    with open("activation_functions.txt", "w") as f:
        f.write(f"Results of the activation functions\n")
        f.write("Batch_size: 32. n_epochs: 2000. HiddenLayers: [50,35]\n")
        f.write(f"Sigmoid Activation Function\n")
        f.write(f'Mean Squared Error: {mseSIG}\n')
        f.write(f'R² Score: {r2SIG}\n')
        f.write(f"ReLU Activation Function\n")
        f.write(f'Mean Squared Error: {mseRE}\n')
        f.write(f'R² Score: {r2RE}\n')
        f.write(f"Leaky ReLU Activation Function\n")
        f.write(f'Mean Squared Error: {mseLRE}\n')
        f.write(f'R² Score: {r2LRE}\n')


if __name__ == "__main__":
    #sigmoid_test()
    # relu_test()
    # leaky_relu_test()
    # all_off_them()    
    prep()