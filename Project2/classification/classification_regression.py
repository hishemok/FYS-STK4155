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
from regression.Regression import Regression

def prep():
    data = read_data()
    #last row is nan and first row is the id
    data = data[:,1:-1]
    #in the dataset 'M' and 'B' are used to represent the classes, we will change them to 1 and 0
    data[data == 'M'] = 1
    data[data == 'B'] = 0
    return data

# def accuracy(predictions, targets):
#     one_hot_predictions = np.zeros(predictions.shape)
#     print(one_hot_predictions.shape)
#     for i, prediction in enumerate(predictions):
#         one_hot_predictions[i, np.argmax(prediction)] = 1
#     return accuracy_score(one_hot_predictions, targets)


if __name__ == "__main__":
    data = prep()
    X = data[:, 1:]  # Features
    y = data[:, 0].astype(int)   # Labels (0 for benign, 1 for malignant)

    scaler = StandardScaler()
    inputs = scaler.fit_transform(X)



    lambda_values = [0,0.0001,0.001,0.01,0.1,1]
    gamma_values = [0,0.0001,0.001,0.01,0.1,1]
    learning_rate_values = [None,0.0001,0.001,0.01,0.1,1]

    best_params = []
    autograd = True
    batch_size = 100
    n_iter = 1000
    test_size = 0.3

    for lmbda in lambda_values:
        print(f"Lambda: {lmbda}")
        for gamma in gamma_values:
                reg = Regression(z = y,dmatrix = inputs)

                beta,z_tilde,z_test = reg.SGD(batch_size=batch_size,n_iterations=n_iter,use_autograd = autograd,Adagrad = True,RMSprop = False,Adam = False,test_size=test_size,lmbda = lmbda,gamma = gamma)
                z_tilde_binary = (z_tilde > 0.5).astype(int)
                accAdagrad = accuracy_score(z_tilde_binary, z_test)
                results = {"Optimizer:":"Adagrad","lambda":lmbda,"gamma":gamma,"accuracy":accAdagrad}

                best_params.append(results)

                beta,z_tilde,z_test = reg.SGD(batch_size=batch_size,n_iterations=n_iter,use_autograd = autograd,Adagrad = False,RMSprop = True,Adam = False,test_size=test_size,lmbda = lmbda,gamma = gamma)
                z_tilde_binary = (z_tilde > 0.5).astype(int)
                accRMSprop = accuracy_score(z_tilde_binary, z_test)
                results = {"Optimizer:":"RMSprop","lambda":lmbda,"gamma":gamma,"accuracy":accRMSprop}

                best_params.append(results)

                beta,z_tilde,z_test = reg.SGD(batch_size=batch_size,n_iterations=n_iter,use_autograd = autograd,Adagrad = False,RMSprop = False,Adam = True,test_size=test_size,lmbda = lmbda,gamma = gamma)
                z_tilde_binary = (z_tilde > 0.5).astype(int)
                accAdam = accuracy_score(z_tilde_binary, z_test)
                results = {"Optimizer:":"Adam","lambda":lmbda,"gamma":gamma,"accuracy":accAdam}

                best_params.append(results)

                #write to file
                df = pd.DataFrame(best_params)
                df.to_csv("classification_results_fixedLR.csv")






























    # reg = Regression(0,0,y,0,X)

    # """Hessian learning rate, no regularization"""
    # reg = Regression(z = y,dmatrix = inputs)
    # beta,z_tilde,z_test = reg.SGD(batch_size=100,n_iterations=1000,test_size=0.3)
    # z_tilde_binary = (z_tilde > 0.5).astype(int)
    # acc = accuracy_score(z_tilde_binary, z_test)
    # print("No optimization: ",acc)

    # #Find optimal batch size and iterations:
    # # batch_sizes = [10,25,50,100]
    # # n_iterations = [500,1000,5000]
    # # for batch_size in batch_sizes:
    # #     for n_iter in n_iterations:
    # #         reg = Regression(z = y,dmatrix = inputs)
    # #         beta,z_tilde,z_test = reg.SGD(batch_size=batch_size,n_iterations=n_iter)
    # #         z_tilde_binary = (z_tilde > 0.5).astype(int)
    # #         acc = accuracy_score(z_tilde_binary, z_test)
    # #         print(f"Batch size: {batch_size}, Iterations: {n_iter}, Accuracy: {acc}")

    # """Batch 100 iterations 1000"""

    # # exit()
    # batch_size = 100
    # n_iterations = 1000



    # """Hessian learning rate, with Adagrad"""
    # reg = Regression(z = y,dmatrix = inputs)
    # beta,z_tilde,z_test = reg.SGD(batch_size=batch_size,n_iterations=n_iterations,Adagrad = True,test_size=0.3)
    # z_tilde_binary = (z_tilde > 0.5).astype(int)
    # acc = accuracy_score(z_tilde_binary, z_test)
    # print("Adagrad optimization: ",acc)

    # """Hessian learning rate, with RMSprop"""
    # reg = Regression(z = y,dmatrix = inputs)
    # beta,z_tilde,z_test = reg.SGD(batch_size=batch_size,n_iterations=n_iterations,RMSprop = True,test_size=0.3)
    # z_tilde_binary = (z_tilde > 0.5).astype(int)
    # acc = accuracy_score(z_tilde_binary, z_test)
    # print("RMSprop optimization: ",acc)

    # """Hessian learning rate, with Adam"""
    # reg = Regression(z = y,dmatrix = inputs)
    # beta,z_tilde,z_test = reg.SGD(batch_size=batch_size,n_iterations=n_iterations,Adam = True,test_size=0.3)
    # z_tilde_binary = (z_tilde > 0.5).astype(int)
    # acc = accuracy_score(z_tilde_binary, z_test)
    # print("Adam optimization: ",acc)

    # """Autograd learning rate, no regularization"""

    # reg = Regression(z = y,dmatrix = inputs)
    # beta,z_tilde,z_test = reg.SGD(batch_size=batch_size,n_iterations=n_iterations,use_autograd = True,test_size=0.3)
    # z_tilde_binary = (z_tilde > 0.5).astype(int)
    # acc = accuracy_score(z_tilde_binary, z_test)
    # print("Autograd optimization: ",acc)

    # """No optimization:  0.7894736842105263
    # Adagrad optimization:  0.7631578947368421
    # RMSprop optimization:  0.8245614035087719
    # Adam optimization:  0.7456140350877193
    # Autograd optimization:  0.7982456140350878"""

    # """autograd + RMSprop is the best combination"""
    # reg = Regression(z = y,dmatrix = inputs)
    # beta,z_tilde,z_test = reg.SGD(batch_size=batch_size,n_iterations=n_iterations,use_autograd = True,Adam = True,test_size=0.3)
    # z_tilde_binary = (z_tilde > 0.5).astype(int)
    # acc = accuracy_score(z_tilde_binary, z_test)
    # print("Autograd + RMSprop optimization: ",acc)
    