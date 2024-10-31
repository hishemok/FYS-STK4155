import autograd.numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
np.random.seed(0)
from sklearn.preprocessing import StandardScaler

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.frankefunction import franke_function


def ReLU(z):
    return np.maximum(0, z)

def ReLU_derivative(z):
    return np.where(z > 0, 1, 0)

def leaky_ReLU(z):
    return np.where(z > 0, z, 0.01 * z)

def leaky_ReLU_derivative(z):
    return np.where(z > 0, 1, 0.01)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def MSE(pred, targets):
    return np.mean((pred - targets) ** 2)

def MSE_derivative(pred, targets):
    return 2 * (pred - targets) / len(pred)

def R2(pred, targets):    
    return 1 - np.sum((targets - pred) ** 2) / np.sum((targets - np.mean(targets)) ** 2)

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, hidden_layer_activation_function = sigmoid, activation_der = sigmoid_derivative,output_layer_activation_functions = lambda x: x,cost_function = MSE, cost_der = MSE_derivative):
        self.layers = []
        self.activation_funcs = []
        self.losses = []
        self.activation_der =[]#activation_der
        self.cost_function = cost_function
        self.cost_der = cost_der


        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_derivative = activation_der
        self.hidden_layer_activation_function = hidden_layer_activation_function
        self.output_layer_activation_functions = output_layer_activation_functions

        self.initialize_layers()

    def initialize_layers(self):
        # Create the layers
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1])# * 0.01
            b = np.zeros((1, layer_sizes[i + 1]))
            self.layers.append((W, b))

            if i < len(layer_sizes) - 2:  # Use ReLU for hidden layers
                if  isinstance(self.hidden_layer_activation_function, list):
                    self.activation_funcs.append(self.hidden_layer_activation_function[i])
                    self.activation_der = self.activation_derivative[i]
                    
                else:
                    self.activation_funcs.append(self.hidden_layer_activation_function)
                    self.activation_der = self.activation_derivative
                    
            else:  # Linear for output layer
                self.activation_funcs.append(self.output_layer_activation_functions)  #

    def reset_weights(self):
        self.initialize_layers()

    def forward(self, inputs):
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = np.dot(a, W) + b
            a = activation_func(z)
        return a
    
    def feed_forward_saver(self, inputs):
        a = inputs
        activations = [a]
        zs = []
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = np.dot(a, W) + b
            zs.append(z)
            a = activation_func(z)
            activations.append(a)
        return activations, zs
    
    def backward(self, inputs, targets):
        gradients = []
        activations, zs = self.feed_forward_saver(inputs)

        dC_da = self.cost_der(activations[-1],targets)

        for i in reversed(range(len(self.layers))):
            W,b = self.layers[i]

            dW = np.dot(activations[i].T, dC_da)
            db = np.sum(dC_da, axis=0, keepdims=True)
            gradients.append((dW, db))

            if i > 0:
                dC_da = np.dot(dC_da, W.T) * self.activation_der(zs[i - 1])
        
        return gradients[::-1]

    
    def update_weights(self, gradients, learning_rate):
        for i in range(len(self.layers)):
            W, b = self.layers[i]
            gradients = [(g[0].astype(np.float64), g[1].astype(np.float64)) for g in gradients]
            W -= learning_rate * gradients[i][0]
            b -= learning_rate * gradients[i][1]
            self.layers[i] = (W, b)

    def train(self, inputs, targets, n_epochs, learning_rate, batch_size):
        n_samples = inputs.shape[0]
        for epoch in range(n_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            inputs = inputs[indices]
            targets = targets[indices]

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = inputs[start:end]
                y_batch = targets[start:end]

                preds = self.forward(X_batch)
                gradients = self.backward(X_batch, y_batch)
                self.update_weights(gradients, learning_rate)

            # Calculate loss on the full dataset for logging
            preds_full = self.forward(inputs)
            loss = self.cost_function(preds_full, targets)

            self.losses.append(loss)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')







if __name__ == "__main__":
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
    # inputs = np.hstack((x,y))
    print(inputs.shape,z.shape)

    nn = NeuralNetwork(input_size=2, hidden_sizes=[50, 25], output_size=1)
    nn.train(inputs, z.reshape(-1, 1), n_epochs=2000, learning_rate=0.001, batch_size=32) 

    preds = nn.forward(inputs)  
    mse = MSE(preds, z)
    r2 = R2(preds, z)

    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')

    print(x.shape,y.shape,z.shape,preds.shape)


    # Plot MSE scores vs epochs
    plt.figure()
    plt.plot(range(len(nn.losses)), nn.losses)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs Epochs')
    plt.grid()
    plt.show()




    sorted_indices = np.lexsort((x.flatten(), y.flatten()))
    x_sorted = x[sorted_indices].flatten()
    y_sorted = y[sorted_indices].flatten()
    z_sorted = z[sorted_indices].flatten()
    preds_sorted = preds[sorted_indices].flatten()

    # Define grid size and reshape
    grid_size = int(np.sqrt(len(x_sorted)))
    xx = x_sorted.reshape((grid_size, grid_size))
    yy = y_sorted.reshape((grid_size, grid_size))
    zz = z_sorted.reshape((grid_size, grid_size))
    preds_reshaped = preds_sorted.reshape((grid_size, grid_size))

    # Plotting
    fig = plt.figure(figsize=(12, 5))

    # Franke function plot
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(xx, yy, zz, cmap='viridis', edgecolor='none')
    ax.set_title("Franke Function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Neural network approximation plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(xx, yy, preds_reshaped, cmap='viridis', edgecolor='none')
    ax2.set_title("Neural Network Approximation")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("preds")

    plt.tight_layout()
    plt.show()
        