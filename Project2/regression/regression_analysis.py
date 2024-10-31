import autograd.numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error as RMSE
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.frankefunction import franke_function
from regression.Regression import Regression


def Polynomial_degree():
    """franke function"""
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    xm, ym = np.meshgrid(x, y)
    z = franke_function(xm, ym, noise=0.0)
    z_flat = z.flatten()

    #First find the optimal polynomial degree for GD and SGD on Franke function
    #and compare with OLS
    polynomial_degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    mse_vals = np.zeros((3,len(polynomial_degrees)))
    r2_vals = np.zeros_like(mse_vals)
    for degree in polynomial_degrees:
        print("Polynomial degree: ",degree)
        reg = Regression(x,y,z_flat,degree)
        X = reg.X

        #OLS
        beta_ols,z_tilde_ols,z_test_ols = reg.linear_regression()
        mse_vals[0,degree-1] = reg.MSE(z_test_ols,z_tilde_ols)
        r2_vals[0,degree-1] = reg.R2(z_test_ols,z_tilde_ols)

        #GD
        beta_gd,z_tilde_gd,z_test_gd = reg.GD(n_iterations=1000)
        mse_vals[1,degree-1] = reg.MSE(z_test_gd,z_tilde_gd)
        r2_vals[1,degree-1] = reg.R2(z_test_gd,z_tilde_gd)

        #SGD
        beta_sgd,z_tilde_sgd,z_test_sgd = reg.SGD(batch_size=25,n_iterations=1000)
        mse_vals[2,degree-1] = reg.MSE(z_test_sgd,z_tilde_sgd)
        r2_vals[2,degree-1] = reg.R2(z_test_sgd,z_tilde_sgd)

    print("MSE values for OLS, GD and SGD for different polynomial degrees:")
    print(mse_vals)
    print("R2 values for OLS, GD and SGD for different polynomial degrees:")
    print(r2_vals)

    #Plotting
    fig,ax = plt.subplots(2,1,figsize=(10,10))
    ax[0].plot(polynomial_degrees,mse_vals[0,:],label='OLS')
    ax[0].plot(polynomial_degrees,mse_vals[1,:],label='GD')
    ax[0].plot(polynomial_degrees,mse_vals[2,:],label='SGD')
    ax[0].set_xlabel('Polynomial degree')
    ax[0].set_ylabel('MSE')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(polynomial_degrees,r2_vals[0,:],label='OLS')
    ax[1].plot(polynomial_degrees,r2_vals[1,:],label='GD')
    ax[1].plot(polynomial_degrees,r2_vals[2,:],label='SGD')
    ax[1].set_xlabel('Polynomial degree')
    ax[1].set_ylabel('R2')
    ax[1].legend()
    ax[1].grid()

    ax[0].set_title('MSE vs Polynomial degree')
    ax[1].set_title('R2 vs Polynomial degree')
    plt.savefig('Polynomial_degree.png')

    plt.show()

    #write to file

    with open('Polynomial_degree.txt','w') as f:
        f.write('GD and SGD no optimization, no parameter tuning. Batches for SGD = 25, n_iter=1000 for both\n')
        f.write("MSE values for OLS, GD and SGD for different polynomial degrees:\n")
        f.write(str(mse_vals.T))
        f.write("\n")
        f.write("R2 values for OLS, GD and SGD for different polynomial degrees:\n")
        f.write(str(r2_vals.T))
        f.write("\n")



def inspect_learning_rate():
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    xm, ym = np.meshgrid(x, y)
    z = franke_function(xm, ym, noise=0.0)
    z_flat = z.flatten()

    polynomial_degree = 10

    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]

    mse_vals = np.zeros((2, len(learning_rates)))
    r2_vals = np.zeros_like(mse_vals)

    for i, learning_rate in enumerate(learning_rates):
        print("Learning rate: ", learning_rate)
        reg = Regression(x, y, z_flat, polynomial_degree)
        X = reg.X

        # GD
        beta_gd, z_tilde_gd, z_test_gd = reg.GD(n_iterations=1000, learning_rate=learning_rate)
        mse_vals[0, i] = reg.MSE(z_test_gd, z_tilde_gd)
        r2_vals[0, i] = reg.R2(z_test_gd, z_tilde_gd)

        # SGD
        beta_sgd, z_tilde_sgd, z_test_sgd = reg.SGD(batch_size=25, n_iterations=1000, learning_rate=learning_rate)
        mse_vals[1, i] = reg.MSE(z_test_sgd, z_tilde_sgd)
        r2_vals[1, i] = reg.R2(z_test_sgd, z_tilde_sgd)


    print("MSE values for GD and SGD for different learning rates:")
    print(mse_vals)
    print("R2 values for GD and SGD for different learning rates:")
    print(r2_vals)

    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(learning_rates, mse_vals[0, :], label='GD')
    ax[0].plot(learning_rates, mse_vals[1, :], label='SGD')
    ax[0].set_xlabel('Learning rate')
    ax[0].set_ylabel('MSE')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(learning_rates, r2_vals[0, :], label='GD')
    ax[1].plot(learning_rates, r2_vals[1, :], label='SGD')
    ax[1].set_xlabel('Learning rate')
    ax[1].set_ylabel('R2')
    ax[1].legend()
    ax[1].grid()

    ax[0].set_title('MSE vs Learning rate')
    ax[1].set_title('R2 vs Learning rate')
    plt.savefig('Learning_rate.png')

    plt.show()

    # write to file

    with open('Learning_rate.txt', 'w') as f:
        f.write('GD and SGD no optimization, no parameter tuning. Batches for SGD = 25, n_iter=1000 for both\n')
        f.write(f"Learning: rates {learning_rates}\n")
        f.write("MSE values for GD and SGD for different learning rates:\n")
        f.write(str(mse_vals.T))
        f.write("\n")
        f.write("R2 values for GD and SGD for different learning rates:\n")
        f.write(str(r2_vals.T))
        f.write("\n")



def inspect_momentum():
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    xm, ym = np.meshgrid(x, y)
    z = franke_function(xm, ym, noise=0.0)
    z_flat = z.flatten()

    polynomial_degree = 10

    momentums = [0.0, 0.5, 0.9, 0.99]

    mse_vals = np.zeros((2, len(momentums)))
    r2_vals = np.zeros_like(mse_vals)

    for i, momentum in enumerate(momentums):
        print("Momentum: ", momentum)
        reg = Regression(x, y, z_flat, polynomial_degree)
        X = reg.X

        # GD
        beta_gd, z_tilde_gd, z_test_gd = reg.GD(n_iterations=1000, gamma=momentum)
        mse_vals[0, i] = reg.MSE(z_test_gd, z_tilde_gd)
        r2_vals[0, i] = reg.R2(z_test_gd, z_tilde_gd)

        # SGD
        beta_sgd, z_tilde_sgd, z_test_sgd = reg.SGD(batch_size=25, n_iterations=1000, gamma=momentum)
        mse_vals[1, i] = reg.MSE(z_test_sgd, z_tilde_sgd)
        r2_vals[1, i] = reg.R2(z_test_sgd, z_tilde_sgd)

    print("MSE values for GD and SGD for different momentums:")
    print(mse_vals)
    print("R2 values for GD and SGD for different momentums:")
    print(r2_vals)

    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(momentums, mse_vals[0, :], label='GD')
    ax[0].plot(momentums, mse_vals[1, :], label='SGD')
    ax[0].set_xlabel('Momentum')
    ax[0].set_ylabel('MSE')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(momentums, r2_vals[0, :], label='GD')
    ax[1].plot(momentums, r2_vals[1, :], label='SGD')
    ax[1].set_xlabel('Momentum')
    ax[1].set_ylabel('R2')
    ax[1].legend()
    ax[1].grid()

    ax[0].set_title('MSE vs Momentum')
    ax[1].set_title('R2 vs Momentum')
    plt.savefig('Momentum.png')

    plt.show()

    # write to file

    with open('Momentum.txt', 'w') as f:
        f.write('GD and SGD no optimization, no parameter tuning. Batches for SGD = 25, n_iter=1000 for both\n')
        f.write(f"Momentums: {momentums}\n")
        f.write("MSE values for GD and SGD for different momentums:\n")
        f.write(str(mse_vals.T))
        f.write("\n")
        f.write("R2 values for GD and SGD for different momentums:\n")
        f.write(str(r2_vals.T))
        f.write("\n")


def COMBINATION():
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    xm, ym = np.meshgrid(x, y)
    z = franke_function(xm, ym, noise=0.0)
    z_flat = z.flatten()

    polynomial_degree = 10

    reg = Regression(x, y, z_flat, polynomial_degree)

    mse_gd = 0
    r2_gd = 0
    mse_sgd = 0
    r2_sgd = 0

    # GD
    beta_gd, z_tilde_gd, z_test_gd = reg.GD(n_iterations=1000, learning_rate=0.1, gamma=0.99)
    mse_gd = reg.MSE(z_test_gd, z_tilde_gd)
    r2_gd = reg.R2(z_test_gd, z_tilde_gd)

    # SGD
    beta_sgd, z_tilde_sgd, z_test_sgd = reg.SGD(batch_size=25, n_iterations=1000, learning_rate=1, gamma=0.99)

    mse_sgd = reg.MSE(z_test_sgd, z_tilde_sgd)
    r2_sgd = reg.R2(z_test_sgd, z_tilde_sgd)

    print("MSE values for GD and SGD for different momentums:")
    print(mse_gd, mse_sgd)
    print("R2 values for GD and SGD for different momentums:")
    print(r2_gd, r2_sgd)


def COMBINATION2():
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    xm, ym = np.meshgrid(x, y)
    z = franke_function(xm, ym, noise=0.1)
    z_flat = z.flatten()

    polynomial_degree = 4

    reg = Regression(x, y, z_flat, polynomial_degree)

    n = 50
    learning_rates = np.linspace(0.02, 0.2, n)
    momentums = np.linspace(0.8, 0.95, n)

    mse_array = np.zeros((len(learning_rates), len(momentums)))

    # SGD
    saved = False
    if saved:
        mse_array = np.load("mse_array.npy")
    else:
        for i, learning_rate in enumerate(tqdm(learning_rates)):
            for j, momentum in enumerate(momentums):
                beta_gd, z_tilde_gd, z_test_gd = reg.GD(n_iterations=100, learning_rate=learning_rate, gamma=momentum)
                mse_array[i, j] = RMSE(z_test_gd, z_tilde_gd)**2
        # np.save("mse_array.npy", mse_array)
    
    print(mse_array)
    # plt.imshow(mse_array, cmap='hot', interpolation='nearest', norm="log")
    plt.contourf(momentums, learning_rates, mse_array, levels=n*n, norm="symlog", cmap='hot')
    plt.colorbar()
    plt.xlabel('Momentum')
    plt.ylabel('Learning rate')
    plt.title('MSE for SGD')
    # plt.savefig('SGD_mse_LR_MOM_Tuned.png')
    plt.show()

def adagrad_momentum():
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    xm, ym = np.meshgrid(x, y)
    z = franke_function(xm, ym, noise=0.0)
    z_flat = z.flatten()

    polynomial_degree = 10

    reg = Regression(x, y, z_flat, polynomial_degree)

    n = 5
    momentum = np.linspace(0.9940, 0.9990, n)

    mse_array = np.zeros((n))
    for i, mom in enumerate(tqdm(momentum)):
        beta_gd, z_tilde_gd, z_test_gd = reg.SGD(batch_size=25,n_iterations=1000, gamma=mom,Adagrad=True)
        mse_array[i] = reg.MSE(z_test_gd, z_tilde_gd)
    
    print(mse_array)
    plt.plot(momentum, mse_array)
    plt.xlabel('Momentum')
    plt.ylabel('MSE')
    plt.title('MSE for SGD with Adagrad')
    # plt.savefig('SGD_mse_momentum_Adagrad.png')
    plt.show()


def iterations_batches():
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    xm, ym = np.meshgrid(x, y)
    z = franke_function(xm, ym, noise=0.0)
    z_flat = z.flatten()

    polynomial_degree = 10

    reg = Regression(x, y, z_flat, polynomial_degree)

    batch_sizes = [5, 10, 25, 50, 100]
    n_iterations = [100, 500, 1000]

    mse_array = np.zeros((len(batch_sizes), len(n_iterations)))

    saved = True
    if saved:
        mse_array = np.load("mse_array.npy")
    else:

        for i, batch_size in enumerate(batch_sizes):
            for j, n_iter in enumerate(n_iterations):
                print("Batch size: ", batch_size, "Iterations: ", n_iter)


                beta_sgd, z_tilde_sgd, z_test_sgd = reg.SGD(batch_size=batch_size, n_iterations=n_iter)
                mse_array[i, j] = reg.MSE(z_test_sgd, z_tilde_sgd)
        np.save("mse_array.npy", mse_array)
    
    print("MSE values for GD and SGD for different batch sizes and iterations:")
    print(mse_array)

    # Plotting subplot heatmatps
    plt.contourf(n_iterations, batch_sizes, mse_array, levels=len(batch_sizes)*len(n_iterations), cmap='hot')
    plt.colorbar()
    plt.xlabel('Iterations')
    plt.ylabel('Batch size')
    plt.title('MSE for SGD | Batch size vs Iterations')
    plt.savefig('SGD_mse_batch_iter.png')
    plt.show()




if __name__ == "__main__":
    #Polynomial_degree()
    #inspect_learning_rate()
    #inspect_momentum()
    # COMBINATION()
    # COMBINATION2()
    # adagrad_momentum()
    iterations_batches()