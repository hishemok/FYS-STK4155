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
from sklearn.model_selection import train_test_split

from nn.neural_network import NeuralNetwork, ReLU, ReLU_derivative, leaky_ReLU, leaky_ReLU_derivative, sigmoid, sigmoid_derivative, MSE, MSE_derivative, R2,softmax
from regression.Regression import Regression
from data.dataloader import read_data

def prep():
    data = read_data()
    #last row is nan and first row is the id
    data = data[:,1:-1]
    #in the dataset 'M' and 'B' are used to represent the classes, we will change them to 1 and 0
    data[data == 'M'] = 1
    data[data == 'B'] = 0
    return data

#plot the data
data = prep()
X = data[:, 1:]  # Features
y = data[:, 0].astype(int)   # Labels (0 for benign, 1 for malignant)

#plot y as heatmap of either 1 or 0

# Calculate approximate grid shape



"""
Hidden Activation Functions                ['softmax', 'ReLU']
Hidden Derivatives             ['<lambda>', 'ReLU_derivative']
Output Activation Function                                ReLU
Cost Function                             binary_cross_entropy
Cost Derivative                                 MSE_derivative
Accuracy (%)                                         99.122807"""


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def binary_cross_entropy(predict, target):
    # Clip values to avoid log(0)
    predict = np.clip(predict, 1e-10, 1 - 1e-10)
    return -np.mean(target * np.log(predict) + (1 - target) * np.log(1 - predict))
hidden_size = [10,5]
learning_rate = 0.1
batch_size = 100
hidden_functions = [softmax, ReLU]
hidden_derivatives = [lambda x: 1, ReLU_derivative]
out = sigmoid
cost_func = binary_cross_entropy
cost_der = MSE_derivative

nn = NeuralNetwork(input_size=30, hidden_sizes=hidden_size, output_size=1,
                            hidden_layer_activation_function=hidden_functions, activation_der=hidden_derivatives,
                            output_layer_activation_functions=out, cost_function=cost_func, cost_der=cost_der)
                    
nn.train(X_train, y_train.reshape(-1, 1), n_epochs=200, learning_rate=learning_rate, batch_size=batch_size)

preds = nn.forward(X)
preds = (preds > 0.5).astype(int)


inputs = X
reg = Regression(z = y,dmatrix = inputs)

beta,z_tilde,z_test = reg.SGD(batch_size=100,n_iterations=1000,use_autograd =True,Adagrad = True)
z_tilde_binary = (z_tilde > 0.5).astype(int)
accAdagrad = accuracy_score(z_tilde_binary, z_test)

sgdpreds = beta @ inputs.T




sgdpreds = (sgdpreds > 0.5).astype(int)

print("shapes: ",y.shape,preds.shape,sgdpreds.shape)



grid_size = int(np.ceil(np.sqrt(len(y))))  # Roughly square grid dimensions
y_true = np.pad(y, (0, grid_size**2 - len(y)), mode='constant')  # Pad with zeros if needed

# Reshape and plot
y_true.reshape(grid_size, grid_size)
# Calculate approximate grid shape
y_preds = np.pad(preds.flatten(), (0, grid_size**2 - len(y)), mode='constant')  
y_sgdpreds = np.pad(sgdpreds, (0, grid_size**2 - len(y)), mode='constant')
# fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # Creates a 1x2 grid of subplots

# # Reshape and plot
# y_predsnn = y_preds.reshape(grid_size, grid_size)
# im2 = axs[1].imshow(y_predsnn, cmap='hot', interpolation='nearest')
# axs[1].set_title("Heatmap of Predicted Classes")
# fig.colorbar(im2, ax=axs[1], label="Class (0 = Benign, 1 = Malignant)")

# y_true.reshape(grid_size, grid_size)
# print(y_predsnn.shape,y_true.shape)

# im1 = axs[0].imshow(y_true.reshape(24,24), cmap='hot', interpolation='nearest')
# axs[0].set_title("Heatmap of True Classes")
# fig.colorbar(im1, ax=axs[0], label="Class (0 = Benign, 1 = Malignant)")

# # Plot the predictions (y_predsnn) heatmap

# plt.suptitle("Comparison of True and Predicted Classes")  # Add an overall title
# plt.show()

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot the ground truth (y_true) heatmap
im1 = axs[0].imshow(y_true.reshape(grid_size,grid_size), cmap='hot', interpolation='nearest')
axs[0].set_title("Heatmap of True Classes")
fig.colorbar(im1, ax=axs[0], label="Class (0 = Benign, 1 = Malignant)")

# Plot the neural network predictions (y_preds) heatmap
im2 = axs[1].imshow(y_preds.reshape(grid_size,grid_size), cmap='hot', interpolation='nearest')
axs[1].set_title("Heatmap of NN Predicted Classes")
fig.colorbar(im2, ax=axs[1], label="Class (0 = Benign, 1 = Malignant)")

# Plot the SGD predictions (sgdpreds) heatmap
im3 = axs[2].imshow(y_sgdpreds.reshape(grid_size,grid_size), cmap='hot', interpolation='nearest')
axs[2].set_title("Heatmap of SGD Predicted Classes")
fig.colorbar(im3, ax=axs[2], label="Class (0 = Benign, 1 = Malignant)")

plt.suptitle("Comparison of True, NN Predicted, and SGD Predicted Classes")
plt.show()