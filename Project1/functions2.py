import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.linear_model as Linmod
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

def FrankeFunction(x, y, noisefactor=0.0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + noisefactor*np.random.randn(x.shape[0], y.shape[1])

def create_design_matrix(x: np.ndarray, y: np.ndarray, degree: int):

    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

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

def scale_X(X,minmax=False, skip_intersept=True):
    if minmax:
        if skip_intersept:
            X_scaled = X.copy()
            scaler = MinMaxScaler()
            scaler.fit(X[:,1:])
            X_scaled[:,1:] = scaler.transform(X[:,1:])
        else:
            scaler = MinMaxScaler()
            scaler.fit(X)
            X_scaled = scaler.transform(X)

    else:
        if skip_intersept:
            X_scaled = X.copy()
            scaler = StandardScaler()
            scaler.fit(X[:,1:])
            X_scaled[:,1:] = scaler.transform(X[:,1:])
        else:
            scaler = StandardScaler()
            scaler.fit(X)
            X_scaled = scaler.transform(X)

    return X_scaled,scaler

def scale_target(z,minmax=False):
    if minmax:
        scaler = MinMaxScaler()
        scaler.fit(z)
        z_scaled = scaler.transform(z)
    else:
        scaler = StandardScaler()
        scaler.fit(z)
        z_scaled = scaler.transform(z)
    return z_scaled,scaler

def OLS(X, z, randomstate=42):
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2,random_state=randomstate)
    beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train
    z_pred = X_test @ beta
    z_model = X @ beta
    z_model_train = X_train @ beta
    return z_pred, z_model ,z_model_train ,beta , z_test, z_train

def Ridge(X, z, lmbda, randomstate=42):
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2,random_state=randomstate)
    beta = np.linalg.inv(X_train.T @ X_train + lmbda*np.eye(X_train.shape[1])) @ X_train.T @ z_train
    z_pred = X_test @ beta
    z_model = X @ beta
    z_model_train = X_train @ beta
    return z_pred, z_model,z_model_train ,beta, z_test, z_train

def Lasso(X, z, lmbda, randomstate=42):
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2,random_state=randomstate)
    clf = Linmod.Lasso(alpha=lmbda)
    clf.fit(X_train, z_train)
    beta = clf.coef_
    z_pred = clf.predict(X_test)
    z_model = clf.predict(X)
    z_model_train = clf.predict(X_train)
    return z_pred, z_model,z_model_train ,beta, z_test, z_train

def k_fold_CV(Model ,X, z,lmbda=0, k=5):
    kf = KFold(n_splits=k)
    if Model == 'OLS':
        model = Linmod.LinearRegression()
    elif Model == 'Ridge':
        model = Linmod.Ridge(alpha=lmbda)
    elif Model == 'Lasso':
        model = Linmod.Lasso(alpha=lmbda)
    error = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        z_train, z_test = z[train_index], z[test_index]
        model.fit(X_train,z_train)
        z_pred = model.predict(X_test)
        beta = model.coef_  
        error += np.mean((z_pred - z_test)**2)
    return error/k, beta


def bootstrap(X, z, n_bootstraps=100, batch_size=100):
    #split data into training and test
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2,random_state=42)
    
    z_pred = np.zeros((z_test.shape[0], n_bootstraps))
    mse = np.zeros(n_bootstraps)
    for j in range(n_bootstraps):
        X_, z_ = resample(X_train, z_train)
        indices = np.random.choice(len(X_), size=batch_size, replace=True)
        X_batch = X_[indices]
        z_batch = z_[indices]
        

        beta = np.linalg.inv(X_batch.T @ X_batch) @ X_batch.T @ z_batch
        z_pred[:, j] = X_test @ beta
        z_model = X @ beta  

        mse[j],_ = MSE_R2(z_test, z_pred[:, j])

    error = np.mean(mse)  
    mean_prediction = np.mean(z_pred, axis=1)
    bias = np.mean((z_test - mean_prediction) ** 2)           
    variance = np.mean(np.var(z_pred, axis=1))    


    return error , bias, variance,z_model



def MSE_R2(z, z_pred):
    mse = np.mean((z - z_pred)**2)
    r2 = 1 - np.sum((z - z_pred)**2)/np.sum((z - np.mean(z))**2)
    return mse, r2