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

def R2(pred, targets):    
    return 1 - np.sum((targets - pred) ** 2) / np.sum((targets - np.mean(targets)) ** 2)



class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, hidden_layer_activation_function = sigmoid, activation_der = sigmoid_derivative,output_layer_activation_functions = lambda x: x,cost_function = MSE):
        self.layers = []
        self.activation_funcs = []
        self.losses = []
        self.activation_der = activation_der
        self.cost_function = cost_function

        # Create the layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1])# * 0.01
            b = np.zeros((1, layer_sizes[i + 1]))
            self.layers.append((W, b))

            if i < len(layer_sizes) - 2:  # Use ReLU for hidden layers
                self.activation_funcs.append(hidden_layer_activation_function)
            else:  # Linear for output layer
                self.activation_funcs.append(output_layer_activation_functions)  #



    def forward(self, inputs):
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = np.dot(a, W) + b
            a = activation_func(z)
        return a
    
    def backward(self, inputs, targets):
        gradients = []
        a = inputs
        activations = [a]
        zs = []

        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = np.dot(a, W) + b
            zs.append(z)
            a = activation_func(z)
            activations.append(a)

        delta = (activations[-1] - targets) / targets.shape[0]  # Mean gradient
        for i in reversed(range(len(self.layers))):
            W, b = self.layers[i]
            gradients.append((np.dot(activations[i].T, delta), np.sum(delta, axis=0, keepdims=True)))
            if i > 0: 
                delta = np.dot(delta, W.T) * self.activation_der(zs[i - 1])  

        return gradients[::-1]
    
    def update_weights(self, gradients, learning_rate):
        for i in range(len(self.layers)):
            W, b = self.layers[i]
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

    # 3D Plot of Predictions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color='b', label='True Values', alpha=0.5)
    ax.scatter(x, y, preds, color='r', label='Predicted Values', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Franke Function)')
    ax.set_title('True vs Predicted Values')
    ax.legend()
    plt.show()

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
        

"""








def ReLU(z):
    return np.where(z > 0, z, 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):

    e_z = np.exp(z - np.max(z, axis=0,keepdims=True))
    return e_z / np.sum(e_z, axis=1,keepdims=True)


def softmax_vec(z):

    e_z = np.exp(z - np.max(z))
    

def sigmoid_der(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

def ReLU_der(z):
    return np.where(z > 0, 1, 0)

def mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def R2(predictions, targets):   
    return 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)

def mse_der(predictions, targets):

    return 2 * (predictions - targets) / len(predictions)

def cross_entropy(predict, target):
    predict = np.clip(predict, 1e-15, 1 - 1e-15)
    return np.sum(-target * np.log(predict))


class NeuralNetwork:
    def __init__(self, network_input_size, layer_output_sizes, activation_funcs,activation_ders, cost_fun, cost_der):

        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.layers = self.create_layers()

    def predict(self, inputs):
        a = inputs

        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a@W + b
            a = activation_func(z)
        return a
        

    def cost(self, inputs, targets):
        predict = self.predict(inputs)
        return self.cost_fun(predict, targets)

    def _feed_forward_saver(self, inputs):
        layer_inputs = []
        zs = []
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a
    
    def create_layers(self):
        layers = []

        i_size = self.network_input_size
        for layer_output_size in self.layer_output_sizes:
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(layer_output_size)
            layers.append((W, b))

            i_size = layer_output_size
        return layers

    def compute_gradient(self, inputs, targets):
        layer_inputs, zs, predict = self._feed_forward_saver(inputs)
        layer_grads = [() for layer in self.layers]

        for i in reversed(range(len(self.layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]

            if i == len(self.layers) - 1:
                dC_da = self.cost_der(predict, targets)
            else:
                (W, b) = self.layers[i + 1]
                dC_da = dC_dz @ W.T

            dC_dz = dC_da * activation_der(z)
            dC_dW = dC_dz.T @ layer_input
            dC_db = dC_dz

            layer_grads[i] = (dC_dW.T, dC_db)

        return layer_grads

    def update_weights(self, layer_grads, learning_rate,targets):      
        # layer_grads = self.compute_gradient(inputs, targets)  

        for i, (W_g, b_g) in enumerate(layer_grads):
            W, b = self.layers[i]
            self.layers[i] = (W - learning_rate * W_g, b - learning_rate * b_g)

    def autograd_compliant_predict(self, inputs):
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = np.dot(a,W) + b
            a = activation_func(z) 
        return a

    def autograd_gradient(self,inputs,targets):
        from autograd import grad

        def cost_func_autograd(inputs, targets):
            pred = self.autograd_compliant_predict(inputs)
            return self.cost_fun(pred, targets)

        gradients = grad(cost_func_autograd)(inputs, targets)
        return gradients.T
    python -c "import matplotlib.pyplot as plt; plt.plot(); plt.show()"
            W, b = self.layers[i]
            print(W.shape, W_g.shape)
            self.layers[i] = (W - learning_rate * W_g, b - learning_rate * b_g)



if __name__ == "__main__":


    # Franke function setup
    n_samples = 100
    x = np.linspace(0, 1, n_samples)
    y = np.linspace(0, 1, n_samples)
    xm, ym = np.meshgrid(x, y)
    z = franke_function(xm, ym, noise=0.0)
    z = z.flatten().reshape(-1, 1)  
    inputs = np.array([xm.flatten(),ym.flatten()]).T

    # Adjust activation functions and layer sizes
    activation_funcs = [sigmoid]  *4
    activation_ders = [sigmoid_der]*4



    network_input_size = inputs.shape[1]
    layer_output_sizes = [128,64,32,1]


    nn = NeuralNetwork(
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        mse,
        mse_der,
    )

    learning_rate = 0.01
    n_epochs = 2000
    msevals = []
    r2vals = []
    # Training loop
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print(f"Epoch {epoch}")
        grads = nn.compute_gradient(inputs, z)

        nn.update_weights(grads, learning_rate,z)
        preds = nn.predict(inputs)
        acc = mse(preds, z)
        r2 = R2(preds, z)
        r2vals.append(r2)
        msevals.append(acc)
    print(f"Final accuracy: {msevals[-1]}")
    print(f"Final R2: {r2vals[-1]}")

    # msevals = []
    # r2vals = []
    # # Training loop
    # for epoch in range(n_epochs):
    #     if epoch % 100 == 0:
    #         print(f"Epoch {epoch}")
    #     grads = nn.autograd_gradient(inputs, z)
    #     print("Gradients len ",len(grads))
    #     nn.autograd_update_weights(grads, learning_rate)
    #     preds = nn.autograd_compliant_predict(inputs)
    #     acc = mse(preds, z)
    #     r2 = R2(preds, z)
    #     r2vals.append(r2)
    #     msevals.append(acc)
    # print(f"Final accuracy: {msevals[-1]}")
    # print(f"Final R2: {r2vals[-1]}")

    plt.plot(range(1, n_epochs + 1), msevals)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Training MSE over Epochs')
    plt.show()



    # Grid for plotting
    xx, yy = np.meshgrid(x, y)
    grid_inputs = np.column_stack((xx.flatten(), yy.flatten()))
    preds = nn.predict(grid_inputs).reshape(xx.shape)

    # Plotting the Franke function and NN approximation
    surf = franke_function(xx, yy, noise=0.0)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(xx, yy, surf)
    ax.set_title("Franke Function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(xx, yy, preds)
    ax2.set_title("Neural Network Approximation")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    plt.show()
"""

