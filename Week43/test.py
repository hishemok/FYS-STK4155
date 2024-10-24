import autograd.numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)
    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)


def ReLU(z):
    return np.where(z > 0, z, 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0,keepdims=True))
    return e_z / np.sum(e_z, axis=1,keepdims=True)


def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.P
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def sigmoid_der(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

def ReLU_der(z):
    return np.where(z > 0, 1, 0)



class NeuralNetwork:
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der,
    ):
        self.NIS = network_input_size
        self.LOS = layer_output_sizes
        self.AF = activation_funcs
        self.AFder = activation_ders
        self.CF = cost_fun
        self.CFder = cost_der
        self.layers = self.create_layers()

    def predict(self, inputs):
        a = inputs

        for (W, b), activation_func in zip(self.layers, activation_funcs):
            z = np.dot(a,W) + b
            a = activation_func(z)
        return a
        

    def cost(self, inputs, targets):
        predict = self.predict(inputs)
        return self.CF(predict, targets)

    def _feed_forward_saver(self, inputs):
        layer_inputs = []
        zs = []
        a = inputs
        for (W, b), activation_func in zip(self.layers, activation_funcs):
            layer_inputs.append(a)
            z = np.dot(a,W) + b
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a
    
    def create_layers(self):
        layers = []

        i_size = self.NIS
        for layer_output_size in self.LOS:
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
                dC_da = self.CFder(predict, targets)
            else:
                (W, b) = self.layers[i + 1]
                dC_da = dC_dz @ W.T

            dC_dz = dC_da * activation_der(z)
            dC_dW = np.dot(layer_input.T, dC_dz)
            dC_db = dC_dz.sum(axis=0)

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def update_weights(self, layer_grads, learning_rate,inputs, targets):
        layer_grads = self.compute_gradient(inputs, targets)
        
        for (W, b), (W_g, b_g) in zip(self.layers, layer_grads):
            W -= learning_rate * W_g
            b -= learning_rate * b_g

    # These last two methods are not needed in the project, but they can be nice to have! The first one has a layers parameter so that you can use autograd on it
    def autograd_compliant_predict(self, inputs):
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.AF):
            z = np.dot(a,W) + b
            a = activation_func(z) 
        return a

    def autograd_gradient(self, targets,inputs):
        from autograd import grad

        def cost_func_autograd(inputs, targets):
            pred = self.autograd_compliant_predict(inputs)
            return self.CF(pred, targets)

        gradients = grad(cost_func_autograd)(inputs, targets)
        return gradients
    
    
    


newdata = datasets.load_iris()
_, ax = plt.subplots()
scatter = ax.scatter(newdata.data[:, 0], newdata.data[:, 1], c=newdata.target)
ax.set(xlabel=newdata.feature_names[0], ylabel=newdata.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], newdata.target_names, loc="lower right", title="Classes"
)
plt.show()
data = newdata.data
print(data.shape)
targets = np.zeros((len(data), 3))  
for i, t in enumerate(newdata.target):
    targets[i, t] = 1


network_input_size = data.shape[1]
layer_output_sizes = [8,3]#[network_input_size, 3]
activation_funcs = [sigmoid, softmax]
activation_ders = [sigmoid_der, lambda x: 1]

def mse_der(predict, target):
    return 2 * (predict - target) / len(predict)

def cross_entropy(predict, target):
    predict = np.clip(predict, 1e-15, 1 - 1e-15)
    return np.sum(-target * np.log(predict))

network = NeuralNetwork(
    network_input_size,
    layer_output_sizes,
    activation_funcs,
    activation_ders,
    cross_entropy,
    mse_der,
)

learning_rate = 0.01
epochs = 4000

accuracy_values = []

for epoch in range(epochs):
    grads = network.compute_gradient(data, targets)
    network.update_weights(grads, learning_rate, data, targets)
    preds = network.predict(data)
    acc = accuracy(preds, targets)
    accuracy_values.append(acc)
    if epoch % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {acc}")

plt.plot(range(1, epochs + 1), accuracy_values)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.show()

#again, but using auto grad
network = NeuralNetwork(
    network_input_size,
    layer_output_sizes,
    activation_funcs,
    activation_ders,
    cross_entropy,
    mse_der,
)

accuracy_values = []

for epoch in range(epochs):
    grads = network.autograd_gradient(targets,data)
    network.update_weights(grads, learning_rate, data, targets)
    preds = network.predict(data)
    acc = accuracy(preds, targets)
    accuracy_values.append(acc)
    if epoch % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {acc}")

plt.plot(range(1, epochs + 1), accuracy_values)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.show()