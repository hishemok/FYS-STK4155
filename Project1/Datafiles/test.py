from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)

# Generate data.
n = 50
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
xm, ym = np.meshgrid(x,y)

def FrankeFunction(x, y, noisefactor=0.0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + noisefactor*np.random.randn(x.shape[0], y.shape[1])

z = FrankeFunction(xm, ym)

x_flat = xm.flatten().reshape(-1, 1)
y_flat = ym.flatten().reshape(-1, 1)
z_flat = z.flatten().reshape(-1, 1)

#create a 5 degree poylnomial model for plotting
degree = 5
# def create_design_matrix(x, y, degree):
#     polydegs = sum(i for i in range(1, degree + 2)) # Number of combinations in polynomial
    
#     X = np.ones((len(x), polydegs))
#     column = 0
#     for i in range(degree + 1):
#         for j in range(degree + 1 - i):
#             # Create design matrix with columns 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3, ...
#             input = (x**i * y**j)
#             X[:,column] = input
#             column += 1
#     return X

def create_design_matrix(x: np.ndarray, y: np.ndarray, degree: int):
    """
    Creates the design matrix X.

    Parameters:
    x (np.ndarray): The independent variable(s).
    y (np.ndarray): The independent variable(s).
    degree (int): The degree of the polynomial features.

    Returns:
    np.ndarray: The design matrix X.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N_x = len(x); N_y = len(y)
    print(N_x, N_y)
    l = int((degree+1)*(degree+2)/2)
    X = np.ones((int(N_x*N_y), l))
    
    xx, yy = np.meshgrid(x, y)          # Make a meshgrid to get all possible combinations of x and y values
    xx = xx.flatten()
    yy = yy.flatten()

    idx = 1
    for i in range(1, degree+1):
        for j in range(i+1):
            X[:, idx] = xx**(i-j)*yy**j
            idx += 1

    return X


X = create_design_matrix(x, y, degree)

# Split the data
X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2, random_state=0)

#Calculate MSE and R2 for both scaled and unscaled data
'''
#start with unscaled data | standardscaling(-= mean)
X_train_scaled = X_train.copy()
X_train_scaled[:,1:] = X_train[:,1:] - np.mean(X_train[:,1:], axis=0)

# OLS regression
beta = np.linalg.inv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train

# MSE and R2 score
z_model = X @ beta
mse = mean_squared_error(z_flat, z_model)
r2 = r2_score(z_flat, z_model)
print('5 degree polynomial model')
print(f"MSE standardscaled: {mse:.4f}")
print(f"R2 standardscaled: {r2:.4f}")

#compare with minmaxscaling
X_train_scaled = X_train.copy()
X_train_max = np.max(X_train[:,1:])
X_train_min = np.min(X_train[:,1:])

X_train_scaled[:,1:] = (X_train[:,1:] - X_train_min) / (X_train_max - X_train_min)

# OLS regression
beta = np.linalg.inv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train

# MSE and R2 score
z_model = X @ beta
mse = mean_squared_error(z_flat, z_model)
r2 = r2_score(z_flat, z_model)
print(f"MSE minmaxscaled: {mse:.4f}")
print(f"R2 minmaxscaled: {r2:.4f}")

'''
# OLS regression
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train

# MSE and R2 score
z_model = X @ beta
mse = mean_squared_error(z_flat, z_model)
r2 = r2_score(z_flat, z_model)
print(f"MSE unscaled: {mse:.4f}")
print(f"R2 unscaled: {r2:.4f}")


'''Since there was no significant difference between scaled and unscaled data, I will plot with unscaled data'''
# import ipywidgets as widgets


fig = plt.figure(figsize=(14, 6))
# Original surface
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(xm, ym, z, cmap='viridis')
ax1.set_title("Original Franke Function Surface")
fig.colorbar(surf1, shrink=0.5, aspect=5)

# Predicted surface
z_model = z_model.reshape(n, n)

ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(xm, ym, z_model, cmap='viridis')
ax2.set_title("Predicted Surface using OLS")
fig.colorbar(surf2, shrink=0.5, aspect=5)
plt.show()
