import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from autograd import grad
from tqdm import trange
import inspect
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.frankefunction import franke_function


class Regression:
    def __init__(self,x,y,z,poly_degree):
        """
        Initialize the regression model
        :param x: x-values, must be either a 1D array
        :param y: y-values, must be either a 1D array
        :param z: z-values, must be a 1D array
        :param poly_degree: polynomial degree
        """


        self.x = x
        self.y = y
        self.z = z
        self.poly_degree = poly_degree
        self.X = self.design_matrix()

    def design_matrix(self):
        """
        Create the design matrix with a given polynomial degree
        """
        degree = self.poly_degree
        x = self.x
        y = self.y
        N_x = len(x) 
        N_y = len(y)
        polydegs = (degree + 1)*(degree + 2)//2
        X = np.ones((int(N_x*N_y), polydegs))

        xm, ym = np.meshgrid(x, y)
        xf = xm.flatten()
        yf = ym.flatten()

        column = 1
        for i in range(1, degree + 1):
            for j in range(i + 1):
                X[:, column] = (xf**(i-j))*(yf**j)
                column += 1
    
        return X
    
    def linear_regression(self,lmbda=0.0,test_size=0.2,randomstate=42,x_train = None,x_test = None, z_train = None, z_test = None):
        """
        Linear regression with OLS or Ridge regression
        :param lmbda: Ridge parameter
        """
        X = self.X
        z = self.z

        if x_train is not None and z_train is not None and z_test is not None and x_test is not None:
            x_train = x_train
            z_train = z_train
            x_test = x_test
            z_test = z_test
        else:
            x_train, x_test, z_train, z_test = train_test_split(X, z, test_size=test_size, random_state=randomstate)

        beta = np.linalg.inv(x_train.T @ x_train + lmbda*np.identity(x_train.shape[1])) @ x_train.T @ z_train

        z_tilde = x_test @ beta
        return beta, z_tilde, z_test
    


    def hessian_eigenvalues(self,lmbda=0.0):
        """
        Compute the eigenvalues of the Hessian matrix
        :param lmbda: Ridge parameter
        """
        H = 2 * (self.X.T @ self.X / self.X.shape[0]) + lmbda * np.eye(self.X.shape[1])
        return np.linalg.eigvalsh(H)
    

    def GD(self, n_iterations, learning_rate=None, gamma=0.0, lmbda=0.0, 
       convergence_tol=1e-8, test_size=0.2, randomstate=42, Adagrad=False, 
       RMSprop=False, Adam=False,use_autograd = False,beta_1 =0.9,beta_2=0.999, x_train=None, x_test=None, z_train=None, z_test=None):
        """
        Gradient Descent with adaptive learning rate and momentum.
        :param n_iterations: Number of iterations   
        :param learning_rate: Initial learning rate, default is 1/max(eigenvalues)
        :param gamma: Momentum term, should be between 0 and 1
        :param lmbda: Ridge parameter
        :param convergence_tol: Tolerance level for convergence
        :param test_size: Proportion of data to be used as the test set
        :param randomstate: Random seed for train/test split
        :param Adagrad: Use Adagrad optimization
        :param RMSprop: Use RMSprop optimization
        :param Adam: Use Adam optimization
        :param beta_1: Exponential decay rate for the first moment estimates (Adam), default is 0.9
        :param beta_2: Exponential decay rate for the second moment estimates (Adam), default is 0.999
        :param x_train, x_test, z_train, z_test: Optionally provided train/test data; otherwise, split from self.X and self
        """

        # Use existing train/test split if provided
        if x_train is not None and z_train is not None and z_test is not None and x_test is not None:
            x_train = x_train
            z_train = z_train
            x_test = x_test
            z_test = z_test
        else:
            # Otherwise, create the split
            x_train, x_test, z_train, z_test = train_test_split(self.X, self.z, test_size=test_size, random_state=randomstate)

        # Calculate learning rate if not provided
        if learning_rate is None:
            eigvals = self.hessian_eigenvalues(lmbda=lmbda)
            learning_rate = 1.0 / np.max(eigvals)

        # Initialize parameters
        m, n = x_train.shape
        beta = np.random.rand(n)
        change = 0.0

        
        
        if use_autograd:
            def cost_function(beta):
                regulazation = lmbda * (beta**2).sum()
                res = (x_train @ beta - z_train)**2
                mean = res.mean()
                return mean + regulazation
            autograd_gradient = grad(cost_function)

        # Initialize Adagrad, RMSprop, and Adam variables
        if Adagrad or RMSprop or Adam:
            G = np.zeros(n)
            eps = 1e-8
        if Adam:
            m_t = 0.0#np.zeros(n)
            v_t = 0.0 #np.zeros(n)
            beta1, beta2 = beta_1, beta_2

        # Gradient Descent Loop
        for t in trange(1, n_iterations + 1):
            if use_autograd:
                gradient = autograd_gradient(beta)
            else:
                gradient = (2 / m) * x_train.T @ (x_train @ beta - z_train) + lmbda * beta * 2

            if Adagrad:
                G += gradient ** 2
                new_change = learning_rate / (np.sqrt(G) + eps) * gradient
                beta -= new_change
                change = new_change

            elif RMSprop:
                G += gamma * G + (1 - gamma) * gradient ** 2
                new_change = learning_rate / (np.sqrt(G) + eps) * gradient
                beta -= new_change
                change = new_change

            elif Adam:
                m_t = beta1 * m_t + (1 - beta1) * gradient
                v_t = beta2 * v_t + (1 - beta2) * gradient ** 2

                m_hat = m_t / (1 - beta1 ** t)
                v_hat = v_t / (1 - beta2 ** t)

                new_change = learning_rate / (np.sqrt(v_hat) + eps) * m_hat
                beta -= new_change
                change = new_change

            else:
                new_change = learning_rate * gradient + gamma * change
                beta -= new_change
                change = new_change

            # Check for convergence
            if np.linalg.norm(new_change) < convergence_tol:
                print(f"Convergence reached at iteration {t}")
                break

        # Make predictions on the test set
        z_tilde = x_test @ beta
        return beta, z_tilde, z_test

    
 
    def SGD(self, batch_size, n_iterations, learning_rate=None, gamma=0.0, lmbda = 0.0,convergence_tol=1e-8, test_size=0.2,randomstate=42, Adagrad=False, 
       RMSprop=False, Adam=False,beta_1 =0.9,beta_2=0.999, x_train=None, x_test=None, z_train=None, z_test=None):
        """
        Stochastic Gradient Descent with optional momentum and optimized learning rate.
        
        Parameters:
        - batch_size (int): Size of the mini-batch for each update.
        - n_iterations (int): Number of epochs (full passes over data).
        - learning_rate (float | None): Initial learning rate. If None, it will be computed from Hessian.
        - lmbda (float): Ridge parameter.
        - convergence_tol (float): Tolerance level for convergence.
        - gamma (float): Momentum term, should be between 0 and 1.
        - test_size (float): Proportion of data to be used as the test set.
        - randomstate (int): Random seed for train/test split.
        - x_train, x_test, z_train, z_test: Optionally provided train/test data; otherwise, split from self.X and self.z.
        """
        
        # Load data
        X, z = self.X, self.z
        if x_train is not None and z_train is not None and x_test is not None and z_test is not None:
            x_train, z_train, x_test, z_test = x_train, z_train, x_test, z_test
        else:
            x_train, x_test, z_train, z_test = train_test_split(X, z, test_size=test_size, random_state=randomstate)
        
        m, n = x_train.shape
        beta = np.random.rand(n)
        change = 0.0
        # If no learning rate specified, calculate it based on the Hessian eigenvalues
        if learning_rate is None:
            eigvals = self.hessian_eigenvalues()
            learning_rate = 1.0 / np.max(eigvals)


        # Set momentum parameters
        if gamma < 0.0 or gamma > 1.0:
            print("Momentum term gamma must be between 0.0 and 1.0. Setting gamma to 0.")
            gamma = 0.0

        # Initialize Adagrad, RMSprop, and Adam variables
        if Adagrad or RMSprop or Adam:
            G = np.zeros(n)
            eps = 1e-8
        if Adam:
            m_t = 0.0#np.zeros(n)
            v_t = 0.0 #np.zeros(n)
            beta1, beta2 = beta_1, beta_2

        # Perform SGD with momentum
        for t in trange(1, n_iterations + 1):
            for j in range(m // batch_size):
                random_indices = np.random.randint(0, m, batch_size)
                x_i = x_train[random_indices]
                z_i = z_train[random_indices]
                

                gradient = (2/batch_size)*x_i.T @ (x_i @ beta - z_i) + lmbda * beta * 2  # Include regularization term if provided

                if Adagrad:
                    G += gradient**2
                    new_change = learning_rate / (np.sqrt(G) + eps) * gradient
                    beta -= new_change
                    change = new_change
                
                elif RMSprop:
                    G += gamma * G + (1 - gamma) * gradient ** 2
                    new_change = learning_rate / (np.sqrt(G) + eps) * gradient
                    beta -= new_change
                    change = new_change
                
                elif Adam:
                    m_t = beta1 * m_t + (1 - beta1) * gradient
                    v_t = beta2 * v_t + (1 - beta2) * gradient ** 2

                    m_hat = m_t / (1 - beta1 ** t)
                    v_hat = v_t / (1 - beta2 ** t)

                    new_change = learning_rate / (np.sqrt(v_hat) + eps) * m_hat
                    beta -= new_change
                    change = new_change
                
                else:
                    new_change = learning_rate * gradient + gamma * change
                    beta -= new_change
                    change = new_change
            
            # Convergence check
            if np.linalg.norm(new_change) < convergence_tol:
                print(f"Converged at iteration {t}")
                break

        # Predictions on test set
        z_tilde = x_test @ beta
        return beta, z_tilde, z_test
 
  
    
    def MSE(self,z_tilde,z_test):
        """
        Mean Squared Error
        """
        return np.mean((z_tilde - z_test)**2)

    def R2(self,z_tilde,z_test):
        """
        R2 score
        """
        return 1 - np.sum((z_test - z_tilde)**2)/np.sum((z_test - np.mean(z_test))**2)


    def cross_validation(self, nfolds,model_func, **kwargs):
        """
        Perform cross-validation on the provided regression model function.
        
        :param model_func: The regression model function to evaluate (e.g., self.linear_regression, self.Ridge)
        :param kwargs: Additional keyword arguments specific to the model
        :return: mean MSE and mean R² scores
        """
        kf = KFold(n_splits=nfolds, shuffle=True)
        mse_scores = []
        r2_scores = []

        X_copy = self.X.copy()
        z_copy = self.z.copy()
        for train_index, test_index in kf.split(self.X):

            X_train, X_test = self.X[train_index], self.X[test_index]
            z_train, z_test = self.z[train_index], self.z[test_index]

            #inspect to get the function signature to call the model 
            signature = inspect.signature(model_func)
            #find parameters
            params = {key: kwargs[key] for key in kwargs if key in signature.parameters}
            # params['test_size'] = 1 - (len(test_index) / len(self.X))  # Update test_size 
            params['x_train'] = X_train
            params['x_test'] = X_test
            params['z_train'] = z_train
            params['z_test'] = z_test

            # Train the model with the current fold's training data
            beta, z_tilde, _ = model_func(**params)

            # Evaluate performance
            mse_scores.append(self.MSE(z_tilde, z_test))
            r2_scores.append(self.R2(z_tilde, z_test))
        #Just to make sure the original data is not changed
        self.X = X_copy
        self.z = z_copy
        return mse_scores, r2_scores



    

if __name__ == "__main__":
    """franke function"""
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    xm, ym = np.meshgrid(x, y)
    z = franke_function(xm, ym, noise=0.0)
    z_flat = z.flatten()

    # """Fit models"""
    poly_degree = 10
    z_flat = z.flatten()    
    reg = Regression(x,y,z_flat,poly_degree)
    X = reg.X
    print("OLS Regression")
    beta_ols,z_tilde_ols,z_test_ols = reg.linear_regression()
    print(f"Degree: {poly_degree}, MSE: {reg.MSE(z_tilde_ols,z_test_ols):.4f}, R²: {reg.R2(z_tilde_ols,z_test_ols):.4f}")


    poly_degree = 15
    z_flat = z.flatten()    
    reg = Regression(x,y,z_flat,poly_degree)
    X = reg.X
    print("Ridge Regression")
    beta_ridge,z_tilde_ridge,z_test_ridge = reg.linear_regression(lmbda=0.005)
    print(f"Degree: {poly_degree}, MSE: {reg.MSE(z_tilde_ridge,z_test_ridge):.4f}, R²: {reg.R2(z_tilde_ridge,z_test_ridge):.4f}")


    poly_degree = 8
    z_flat = z.flatten()    
    reg = Regression(x,y,z_flat,poly_degree)
    X = reg.X
    print("Gradient Descent")
    beta_gd,z_tilde_gd,z_test_gd = reg.GD(n_iterations=1000,use_autograd=True)
    print(f"Degree: {poly_degree}, MSE: {reg.MSE(z_tilde_gd,z_test_gd):.4f}, R²: {reg.R2(z_tilde_gd,z_test_gd):.4f}")
    # print("Gradient Descent with Adagrad")
    # beta_gd,z_tilde_gd,z_test_gd = reg.GD(n_iterations=1000,Adagrad=True)
    # print(f"Degree: {poly_degree}, MSE: {reg.MSE(z_tilde_gd,z_test_gd):.4f}, R²: {reg.R2(z_tilde_gd,z_test_gd):.4f}")
    # print("Gradient Descent with Adam")
    # beta_gd,z_tilde_gd,z_test_gd = reg.GD(n_iterations=1000,Adam=True)
    # print(f"Degree: {poly_degree}, MSE: {reg.MSE(z_tilde_gd,z_test_gd):.4f}, R²: {reg.R2(z_tilde_gd,z_test_gd):.4f}")
    # print("Gradient Descent with RMSprop")
    # beta_gd,z_tilde_gd,z_test_gd = reg.GD(n_iterations=1000,RMSprop=True)
    # print(f"Degree: {poly_degree}, MSE: {reg.MSE(z_tilde_gd,z_test_gd):.4f}, R²: {reg.R2(z_tilde_gd,z_test_gd):.4f}")


    poly_degree = 15
    z_flat = z.flatten()    
    reg = Regression(x,y,z_flat,poly_degree)
    X = reg.X
    print("Stochastic Gradient Descent")
    beta_sgd,z_tilde_sgd,z_test_sgd = reg.SGD(batch_size=25,n_iterations=1000)
    print(f"Degree: {poly_degree}, MSE: {reg.MSE(z_tilde_sgd,z_test_sgd):.4f}, R²: {reg.R2(z_tilde_sgd,z_test_sgd):.4f}")
    # print("Stochastic Gradient Descent with Adagrad")
    # beta_sgd,z_tilde_sgd,z_test_sgd = reg.SGD(batch_size=25,n_iterations=1000,Adagrad=True)
    # print(f"Degree: {poly_degree}, MSE: {reg.MSE(z_tilde_sgd,z_test_sgd):.4f}, R²: {reg.R2(z_tilde_sgd,z_test_sgd):.4f}")
    # print("Stochastic Gradient Descent with Adam")
    # beta_sgd,z_tilde_sgd,z_test_sgd = reg.SGD(batch_size=25,n_iterations=1000,Adam=True)
    # print(f"Degree: {poly_degree}, MSE: {reg.MSE(z_tilde_sgd,z_test_sgd):.4f}, R²: {reg.R2(z_tilde_sgd,z_test_sgd):.4f}")
    # print("Stochastic Gradient Descent with RMSprop")
    # beta_sgd,z_tilde_sgd,z_test_sgd = reg.SGD(batch_size=25,n_iterations=1000,RMSprop=True)
    # print(f"Degree: {poly_degree}, MSE: {reg.MSE(z_tilde_sgd,z_test_sgd):.4f}, R²: {reg.R2(z_tilde_sgd,z_test_sgd):.4f}")



    

    



    # """Plot models"""
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(xm, ym, z, cmap='viridis')
    # # ax.plot_surface(xm, ym, z, cmap='viridis')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # #add colorbarz_tilde_ridge.reshape(100,100)
    # plt.colorbar(surf)
    # plt.title('Franke function')
    # plt.show()

    # #3d plot of OLS
    # z_ols = (X@beta_ols).reshape(z.shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(xm, ym, z_ols, cmap='viridis')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # plt.colorbar(surf)
    # plt.title('OLS')
    # plt.show()

    # #3d plot of Ridge
    # z_ridge = (X@beta_ridge).reshape(z.shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(xm, ym, z_ridge, cmap='viridis')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    
    # plt.colorbar(surf)
    # plt.title('Ridge')
    # plt.show()

    # #3d plot of GD
    # z_gd = (X@beta_gd).reshape(z.shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(xm, ym, z_gd, cmap='viridis')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # plt.colorbar(surf)
    # plt.title('GD')
    # plt.show()

    # #3d plot of SGD
    # z_sgd = (X@beta_sgd).reshape(z.shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(xm, ym, z_sgd, cmap='viridis')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # plt.colorbar(surf)
    # plt.title('SGD')if x_train != None and z_train != None and z_test != None:
    # plt.show()

    """Cross-validation"""
    print("\nCross-validation")
    nfolds = 10
    print(f"Number of folds: {nfolds}")


    print("OLS Cross-validation")
    poly_degree = 10
    reg = Regression(x,y,z_flat,poly_degree)
    X = reg.X
    mse_scores_ols, r2_scores_ols = reg.cross_validation(nfolds, reg.linear_regression)


    print("Ridge Cross-validation")
    poly_degree = 15
    reg = Regression(x,y,z_flat,poly_degree)
    X = reg.X
    mse_scores_ridge, r2_scores_ridge = reg.cross_validation(nfolds, reg.linear_regression,lmbda = 1e-5)


    print("GD Cross-validation")
    poly_degree = 8
    reg = Regression(x,y,z_flat,poly_degree)
    X = reg.X
    mse_scores_gd, r2_scores_gd = reg.cross_validation(nfolds, reg.GD, n_iterations=1000,convergence_tol=1e-8)


    print("SGD Cross-validation")
    poly_degree = 15
    z_flat = z.flatten()    
    reg = Regression(x,y,z_flat,poly_degree)
    X = reg.X
    mse_scores_sgd, r2_scores_sgd = reg.cross_validation(nfolds, reg.SGD, batch_size=100, n_iterations=1000,convergence_tol=1e-8, Adam=True) 

    print(f"OLS: MSE = {np.mean(mse_scores_ols):.4f}, R² = {np.mean(r2_scores_ols):.4f}")
    print(f"Ridge: MSE = {np.mean(mse_scores_ridge):.4f}, R² = {np.mean(r2_scores_ridge):.4f}")
    print(f"GD: MSE = {np.mean(mse_scores_gd):.4f}, R² = {np.mean(r2_scores_gd):.4f}")
    print(f"SGD: MSE = {np.mean(mse_scores_sgd):.4f}, R² = {np.mean(r2_scores_sgd):.4f}")

    # Plot MSE and R² scores

    # Plot MSE and R² scores
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # OLS Plot
    ax[0, 0].plot(mse_scores_ols, label='OLS MSE', color='blue')
    ax[0, 0].set_ylabel('MSE', color='blue')
    ax[0, 0].tick_params(axis='y', labelcolor='blue')

    ax_ols_r2 = ax[0, 0].twinx()  # Create a second y-axis for R²
    ax_ols_r2.plot(r2_scores_ols, label='OLS R²', color='orange')
    ax_ols_r2.set_ylabel('R²', color='orange')
    ax_ols_r2.tick_params(axis='y', labelcolor='orange')

    ax[0, 0].set_xlabel('Fold')
    ax[0, 0].set_title('OLS MSE and R²')
    ax[0, 0].legend(loc='upper left')
    ax_ols_r2.legend(loc='upper right')

    # Ridge Plot
    ax[0, 1].plot(mse_scores_ridge, label='Ridge MSE', color='blue')
    ax[0, 1].set_ylabel('MSE', color='blue')
    ax[0, 1].tick_params(axis='y', labelcolor='blue')

    ax_ridge_r2 = ax[0, 1].twinx()  # Create a second y-axis for R²
    ax_ridge_r2.plot(r2_scores_ridge, label='Ridge R²', color='orange')
    ax_ridge_r2.set_ylabel('R²', color='orange')
    ax_ridge_r2.tick_params(axis='y', labelcolor='orange')

    ax[0, 1].set_xlabel('Fold')
    ax[0, 1].set_title('Ridge MSE and R²')
    ax[0, 1].legend(loc='upper left')
    ax_ridge_r2.legend(loc='upper right')

    # GD Plot
    ax[1, 0].plot(mse_scores_gd, label='GD MSE', color='blue')
    ax[1, 0].set_ylabel('MSE', color='blue')
    ax[1, 0].tick_params(axis='y', labelcolor='blue')

    ax_gd_r2 = ax[1, 0].twinx()  # Create a second y-axis for R²
    ax_gd_r2.plot(r2_scores_gd, label='GD R²', color='orange')
    ax_gd_r2.set_ylabel('R²', color='orange')
    ax_gd_r2.tick_params(axis='y', labelcolor='orange')

    ax[1, 0].set_xlabel('Fold')
    ax[1, 0].set_title('GD MSE and R²')
    ax[1, 0].legend(loc='upper left')
    ax_gd_r2.legend(loc='upper right')

    # SGD Plot
    ax[1, 1].plot(mse_scores_sgd, label='SGD MSE', color='blue')
    ax[1, 1].set_ylabel('MSE', color='blue')
    ax[1, 1].tick_params(axis='y', labelcolor='blue')

    ax_sgd_r2 = ax[1, 1].twinx()  # Create a second y-axis for R²
    ax_sgd_r2.plot(r2_scores_sgd, label='SGD R²', color='orange')
    ax_sgd_r2.set_ylabel('R²', color='orange')
    ax_sgd_r2.tick_params(axis='y', labelcolor='orange')

    ax[1, 1].set_xlabel('Fold')
    ax[1, 1].set_title('SGD MSE and R²')
    ax[1, 1].legend(loc='upper left')
    ax_sgd_r2.legend(loc='upper right')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
