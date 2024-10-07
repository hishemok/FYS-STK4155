import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.linear_model as Linmod
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict



def FrankeFunction(x, y, noisefactor=0.0):
    '''
    Inputs x and y are 2D arrays of coordinates.
    '''
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + noisefactor*np.random.randn(x.shape[0], y.shape[1])

def X_design(x, y, degree):
    '''
    Creates the design matrix X.

    Parameters:
    x (np.ndarray): The independent variable(s).
    y (np.ndarray): The independent variable(s).
    degree (int): The degree of the polynomial features.

    Returns:
    np.ndarray: The design matrix X.
    '''
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

def X_scale(X_train,X_test,X ,standard_scaling=True, minmax_scaling=False,skip_intercept=True):
    scaler_X = None  # Initialize scaler variable
    
    if skip_intercept:
        if standard_scaling:
            scaler_X = StandardScaler().fit(X_train[:,1:])
            X_train[:,1:] = scaler_X.transform(X_train[:,1:])
            X_test[:,1:] = scaler_X.transform(X_test[:,1:])
            X[:,1:] = scaler_X.transform(X[:,1:])
        elif minmax_scaling:
            scaler_X = MinMaxScaler().fit(X_train[:,1:])
            X_train[:,1:] = scaler_X.transform(X_train[:,1:])
            X_test[:,1:] = scaler_X.transform(X_test[:,1:])
            X[:,1:] = scaler_X.transform(X[:,1:])
    else:
        if standard_scaling:
            scaler_X = StandardScaler().fit(X_train)
            X_train = scaler_X.transform(X_train)
            X_test = scaler_X.transform(X_test)
            X = scaler_X.transform(X)
        elif minmax_scaling:
            scaler_X = MinMaxScaler().fit(X_train)
            X_train = scaler_X.transform(X_train)
            X_test = scaler_X.transform(X_test)
            X = scaler_X.transform(X)
    return X_train, X_test, X, scaler_X

def target_scale(z_train,z_test,z,standard_scaling=True, minmax_scaling=False):
    scaler_z = None  # Initialize scaler variable
    if standard_scaling:
        scaler_z = StandardScaler().fit(z_train.reshape(-1, 1)) 
        z_train = scaler_z.transform(z_train.reshape(-1, 1)).flatten()
        z_test= scaler_z.transform(z_test.reshape(-1, 1)).flatten()
        z = scaler_z.transform(z.reshape(-1, 1)).flatten()
    elif minmax_scaling:
        scaler_z = MinMaxScaler().fit(z_train.reshape(-1, 1))
        z_train = scaler_z.transform(z_train.reshape(-1, 1)).flatten()
        z_test = scaler_z.transform(z_test.reshape(-1, 1)).flatten()
        z = scaler_z.transform(z.reshape(-1, 1)).flatten()
    return z_train, z_test, z, scaler_z



def OLS(X, z, random_state=0, scaled_data=False,test_size = 0.2, standard_scaling=True, minmax_scaling=False,skip_intercept=True):
    '''
    Parameters:
    X: Design matrix
    z: Target values
    random_state: Seed for random number generator
    test_size: Size of test set. Default is 0.2
    scaled_data: Boolean for scaling data. Default is False
    standard_scaling: Boolean for standard scaling. Default is True
    minmax_scaling: Boolean for minmax scaling. Default is False
    skip_intercept: Boolean for skipping intercept. Default is True


    Returns:
    z_model: Predicted values;
    z_model_train: Predicted values for training set
    z_model_test: Predicted values for test set
    beta: Coefficients
    z_train: Target values for training set
    z_test: Target values for test set
    If standard_scaling or minmax_scaling is True, the function will return the scaled values and the scalers.


    '''
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size, random_state=random_state)

    if scaled_data == False:
        beta,_,_,_ = np.linalg.lstsq(X_train, z_train, rcond=None)
        #beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train
        z_model_train = X_train @ beta
        z_model_test = X_test @ beta
        z_model = X @ beta

        return z_model, z_model_train, z_model_test, beta, z_train, z_test


    if scaled_data:

        if standard_scaling:
            if skip_intercept:
                scaler_X = StandardScaler().fit(X_train[:,1:])
                X_train[:,1:] = scaler_X.transform(X_train[:,1:])
                X_test[:,1:] = scaler_X.transform(X_test[:,1:])
                X[:,1:] = scaler_X.transform(X[:,1:])
            else:
                scaler_X = StandardScaler().fit(X_train)
                X_train = scaler_X.transform(X_train)
                X_test = scaler_X.transform(X_test)
                X = scaler_X.transform(X)

            scaler_z = StandardScaler().fit(z_train.reshape(-1, 1)) 
            z_train = scaler_z.transform(z_train.reshape(-1, 1)).flatten()
            z_test= scaler_z.transform(z_test.reshape(-1, 1)).flatten()

            beta,_,_,_ = np.linalg.lstsq(X_train, z_train, rcond=None)
            #beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train

            z_model_train = X_train @ beta
            z_model_test = X_test @ beta
            z_model = X @ beta

            return z_model, z_model_train, z_model_test, beta, z_train, z_test, scaler_X, scaler_z
        elif minmax_scaling:
            if skip_intercept:
                scaler_X = MinMaxScaler().fit(X_train[:,1:])
                X_train[:,1:] = scaler_X.transform(X_train[:,1:])
                X_test[:,1:] = scaler_X.transform(X_test[:,1:])
                X[:,1:] = scaler_X.transform(X[:,1:])
            else:
                scaler_X = MinMaxScaler().fit(X_train)
                X_train = scaler_X.transform(X_train)
                X_test = scaler_X.transform(X_test)
                X = scaler_X.transform(X)

            scaler_z = MinMaxScaler().fit(z_train.reshape(-1, 1))
            z_train = scaler_z.transform(z_train.reshape(-1, 1)).flatten()
            z_test = scaler_z.transform(z_test.reshape(-1, 1)).flatten()

            #beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train
            beta,_,_,_ = np.linalg.lstsq(X_train, z_train, rcond=None)
            z_model_train = X_train @ beta
            z_model_test = X_test @ beta
            z_model = X @ beta

            return z_model, z_model_train, z_model_test, beta, z_train, z_test, scaler_X, scaler_z       



def Ridge(X, z, lam, random_state=0, scaled_data=False,test_size = 0.2, standard_scaling=True, minmax_scaling=False,skip_intercept=True):
    '''
    Parameters:
    X: Design matrix
    z: Target values
    random_state: Seed for random number generator
    test_size: Size of test set. Default is 0.2
    scaled_data: Boolean for scaling data. Default is False
    standard_scaling: Boolean for standard scaling. Default is True
    minmax_scaling: Boolean for minmax scaling. Default is False
    skip_intercept: Boolean for skipping intercept. Default is True


    Returns:
    z_model: Predicted values;
    z_model_train: Predicted values for training set
    z_model_test: Predicted values for test set
    beta: Coefficients
    z_train: Target values for training set
    z_test: Target values for test set
    If standard_scaling or minmax_scaling is True, the function will return the scaled values and the scalers.


    '''
    if scaled_data == False:
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size, random_state=random_state)

        beta = np.linalg.inv(X_train.T @ X_train + lam*np.eye(X_train.shape[1])) @ X_train.T @ z_train
        z_model_train = X_train @ beta
        z_model_test = X_test @ beta
        z_model = X @ beta

        return z_model, z_model_train, z_model_test, beta, z_train, z_test


    if scaled_data:
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size, random_state=random_state)

        if standard_scaling:
            if skip_intercept:
                scaler_X = StandardScaler().fit(X_train[:,1:])
                X_train[:,1:] = scaler_X.transform(X_train[:,1:])
                X_test[:,1:] = scaler_X.transform(X_test[:,1:])
                X[:,1:] = scaler_X.transform(X[:,1:])
            else:
                scaler_X = StandardScaler().fit(X_train)
                X_train = scaler_X.transform(X_train)
                X_test = scaler_X.transform(X_test)
                X = scaler_X.transform(X)

            scaler_z = StandardScaler().fit(z_train.reshape(-1, 1)) 
            z_train = scaler_z.transform(z_train.reshape(-1, 1)).flatten()
            z_test= scaler_z.transform(z_test.reshape(-1, 1)).flatten()


            beta = np.linalg.inv(X_train.T @ X_train + lam*np.eye(X_train.shape[1])) @ X_train.T @ z_train

            z_model_train = X_train @ beta
            z_model_test = X_test @ beta
            z_model = X @ beta


            return z_model, z_model_train, z_model_test, beta, z_train, z_test, scaler_X, scaler_z
        elif minmax_scaling:
            if skip_intercept:
                scaler_X = MinMaxScaler().fit(X_train[:,1:])
                X_train[:,1:] = scaler_X.transform(X_train[:,1:])
                X_test[:,1:] = scaler_X.transform(X_test[:,1:])
                X[:,1:] = scaler_X.transform(X[:,1:])
            else:
                scaler_X = MinMaxScaler().fit(X_train)
                X_train = scaler_X.transform(X_train)
                X_test = scaler_X.transform(X_test)
                X = scaler_X.transform(X)

            scaler_z = MinMaxScaler().fit(z_train.reshape(-1, 1))
            z_train = scaler_z.transform(z_train.reshape(-1, 1)).flatten()
            z_test = scaler_z.transform(z_test.reshape(-1, 1)).flatten()

            beta = np.linalg.inv(X_train.T @ X_train + lam*np.eye(X_train.shape[1])) @ X_train.T @ z_train

            z_model_train = X_train @ beta
            z_model_test = X_test @ beta
            z_model = X @ beta

            return z_model, z_model_train, z_model_test, beta, z_train, z_test, scaler_X, scaler_z       

def Lasso(X, z, lam, random_state=0, scaled_data=False,test_size = 0.2, standard_scaling=True, minmax_scaling=False,skip_intercept=True):
    '''
    Parameters:
    X: Design matrix
    z: Target values
    random_state: Seed for random number generator
    test_size: Size of test set. Default is 0.2
    scaled_data: Boolean for scaling data. Default is False
    standard_scaling: Boolean for standard scaling. Default is True
    minmax_scaling: Boolean for minmax scaling. Default is False
    skip_intercept: Boolean for skipping intercept. Default is True


    Returns:
    z_model: Predicted values;
    z_model_train: Predicted values for training set
    z_model_test: Predicted values for test set
    beta: Coefficients
    z_train: Target values for training set
    z_test: Target values for test set
    If standard_scaling or minmax_scaling is True, the function will return the scaled values and the scalers.
    '''
    if scaled_data == False:
            X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size, random_state=random_state)

            lasso = Linmod.Lasso(alpha=lam, max_iter=10000)
            lasso.fit(X_train, z_train)

            beta = lasso.coef_
            z_model_train = X_train @ beta
            z_model_test = X_test @ beta
            z_model = X @ beta

            return z_model, z_model_train, z_model_test, beta, z_train, z_test


    if scaled_data:
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size, random_state=random_state)

        if standard_scaling:
            if skip_intercept:
                scaler_X = StandardScaler().fit(X_train[:,1:])
                X_train[:,1:] = scaler_X.transform(X_train[:,1:])
                X_test[:,1:] = scaler_X.transform(X_test[:,1:])
                X[:,1:] = scaler_X.transform(X[:,1:])
            else:
                scaler_X = StandardScaler().fit(X_train)
                X_train = scaler_X.transform(X_train)
                X_test = scaler_X.transform(X_test)
                X = scaler_X.transform(X)

            scaler_z = StandardScaler().fit(z_train.reshape(-1, 1)) 
            z_train = scaler_z.transform(z_train.reshape(-1, 1)).flatten()
            z_test= scaler_z.transform(z_test.reshape(-1, 1)).flatten()


            lasso = Linmod.Lasso(alpha=lam, max_iter=10000)
            lasso.fit(X_train, z_train)

            beta = lasso.coef_

            z_model_train = X_train @ beta
            z_model_test = X_test @ beta
            z_model = X @ beta


            return z_model, z_model_train, z_model_test, beta, z_train, z_test, scaler_X, scaler_z
        elif minmax_scaling:
            if skip_intercept:
                scaler_X = MinMaxScaler().fit(X_train[:,1:])
                X_train[:,1:] = scaler_X.transform(X_train[:,1:])
                X_test[:,1:] = scaler_X.transform(X_test[:,1:])
                X[:,1:] = scaler_X.transform(X[:,1:])
            else:
                scaler_X = MinMaxScaler().fit(X_train)
                X_train = scaler_X.transform(X_train)
                X_test = scaler_X.transform(X_test)
                X = scaler_X.transform(X)

            scaler_z = MinMaxScaler().fit(z_train.reshape(-1, 1))
            z_train = scaler_z.transform(z_train.reshape(-1, 1)).flatten()
            z_test = scaler_z.transform(z_test.reshape(-1, 1)).flatten()

            lasso = Linmod.Lasso(alpha=lam, max_iter=10000)
            lasso.fit(X_train, z_train)

            beta = lasso.coef_

            z_model_train = X_train @ beta
            z_model_test = X_test @ beta
            z_model = X @ beta

            return z_model, z_model_train, z_model_test, beta, z_train, z_test, scaler_X, scaler_z   

def MSE(z, z_model):
    '''
    Parameters:
    z: Target values
    z_model: Predicted values

    Returns:
    MSE: Mean squared error
    '''
    return np.mean((z - z_model)**2)

def R2(z, z_model):
    '''
    Parameters:
    z: Target values
    z_model: Predicted values

    Returns:
    R2: R2 score
    '''
    return 1 - np.sum((z - z_model)**2) / np.sum((z - np.mean(z))**2)


def bias_variance_tradeoff(X, z, n_bootstraps,batch_size, scaling = False,standard_scaling=True, minmax_scaling=False,skip_intercept=True):
    '''
    Parameters:
    X: Design matrix
    z: Target values
    n_bootstraps: Number of bootstraps
    batch_size: Size of batch
    scaling: Boolean for scaling data. Default is False
    standard_scaling: Boolean for standard scaling. Default is True
    minmax_scaling: Boolean for minmax scaling. Default is False
    skip_intercept: Boolean for skipping intercept. Default is True

    Returns:
    error: Mean squared error
    bias: Bias
    variance: Variance

    '''

    # Split the data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=0)
    if scaling == True:
        X_train, X_test, X, scaler_X = X_scale(X_train,X_test,X,standard_scaling=True, minmax_scaling=False,skip_intercept=True)
        z_train, z_test, z, scaler_z= target_scale(z_train,z_test,z,standard_scaling=True, minmax_scaling=False)
    
    z_pred = np.zeros((z_test.shape[0], n_bootstraps))
    mse = np.zeros(n_bootstraps)
    for j in range(n_bootstraps):
        X_, z_ = resample(X_train, z_train)
        indices = np.random.choice(len(X_), size=batch_size, replace=True)
        X_batch = X_[indices]
        z_batch = z_[indices]
        

        beta = np.linalg.inv(X_batch.T @ X_batch) @ X_batch.T @ z_batch
        z_pred[:, j] = X_test @ beta

        mse[j] = MSE(z_test, z_pred[:, j])

    error = np.mean(mse)  
    mean_prediction = np.mean(z_pred, axis=1)
    bias = np.mean((z_test - mean_prediction) ** 2)           
    variance = np.mean(np.var(z_pred, axis=1))    


    return error , bias, variance



def kfold(Model_name: str,X,z,k, random_state=0,scaling = False,standard_scaling=True, minmax_scaling=False,skip_intercept=True, lam=1e-4):
    '''
    Parameters:
    Model_name: Name of the model. Must be either OLS, Ridge or Lasso
    X: Design matrix
    z: Target values
    k: Number of folds
    random_state: Seed for random number generator
    scaling: Boolean for scaling data. Default is False
    standard_scaling: Boolean for standard scaling. Default is True
    minmax_scaling: Boolean for minmax scaling. Default is False
    skip_intercept: Boolean for skipping intercept. Default is True
    lam: Regularization parameter. Default is 1e-4

    Returns:
    z_pred: Predicted values
    error: Mean squared error
    '''




    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    error = np.zeros(k)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=random_state)
    if scaling:
        X_train, X_test, X, scaler_X = X_scale(X_train,X_test,X,standard_scaling=standard_scaling, minmax_scaling=minmax_scaling,skip_intercept=skip_intercept)
        z_train, z_test, z, scaler_z= target_scale(z_train,z_test,z,standard_scaling=standard_scaling, minmax_scaling=minmax_scaling)


    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        z_train, z_test = z[train_index], z[test_index]

        if Model_name == 'OLS':
            beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train
            z_pred = X_test @ beta
            z_full = X @ beta

        elif Model_name == 'Ridge':
            beta = np.linalg.inv(X_train.T @ X_train + lam*np.eye(X_train.shape[1])) @ X_train.T @ z_train
            z_pred = X_test @ beta
            z_full = X @ beta

        elif Model_name == 'Lasso':
            model = Linmod.Lasso(alpha=lam, max_iter=10000)
            model.fit(X_train, z_train)
            beta = model.coef_
            z_pred = X_test @ beta
            z_full = X @ beta

        else:
            raise ValueError('Model_name must be either OLS, Ridge or Lasso')
        
        if scaling:
            z_pred = scaler_z.inverse_transform(z_pred.reshape(-1, 1)).flatten()
            z_full = scaler_z.inverse_transform(z_full.reshape(-1, 1)).flatten()

        error[i] = MSE(z_test, z_pred)
    
    return z_pred , error , z_full


