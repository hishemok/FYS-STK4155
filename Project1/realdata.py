import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
#['magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight', 'twilight_shifted', 'turbo', 'Blues', 'BrBG', 'BuGn', 'BuPu', 'CMRmap', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Spectral', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone', 'brg', 'bwr', 'cool', 'coolwarm', 'copper', 'cubehelix', 'flag', 'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar', 'gist_rainbow', 'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2', 'gray', 'hot', 'hsv', 'jet', 'nipy_spectral', 'ocean', 'pink', 'prism', 'rainbow', 'seismic', 'spring', 'summer', 'terrain', 'winter', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c', 'magma_r', 'inferno_r', 'plasma_r', 'viridis_r', 'cividis_r', 'twilight_r', 'twilight_shifted_r', 'turbo_r', 'Blues_r', 'BrBG_r', 'BuGn_r', 'BuPu_r', 'CMRmap_r', 'GnBu_r', 'Greens_r', 'Greys_r', 'OrRd_r', 'Oranges_r', 'PRGn_r', 'PiYG_r', 'PuBu_r', 'PuBuGn_r', 'PuOr_r', 'PuRd_r', 'Purples_r', 'RdBu_r', 'RdGy_r', 'RdPu_r', 'RdYlBu_r', 'RdYlGn_r', 'Reds_r', 'Spectral_r', 'Wistia_r', 'YlGn_r', 'YlGnBu_r', 'YlOrBr_r', 'YlOrRd_r', 'afmhot_r', 'autumn_r', 'binary_r', 'bone_r', 'brg_r', 'bwr_r', 'cool_r', 'coolwarm_r', 'copper_r', 'cubehelix_r', 'flag_r', 'gist_earth_r', 'gist_gray_r', 'gist_heat_r', 'gist_ncar_r', 'gist_rainbow_r', 'gist_stern_r', 'gist_yarg_r', 'gnuplot_r', 'gnuplot2_r', 'gray_r', 'hot_r', 'hsv_r', 'jet_r', 'nipy_spectral_r', 'ocean_r', 'pink_r', 'prism_r', 'rainbow_r', 'seismic_r', 'spring_r', 'summer_r', 'terrain_r', 'winter_r', 'Accent_r', 'Dark2_r', 'Paired_r', 'Pastel1_r', 'Pastel2_r', 'Set1_r', 'Set2_r', 'Set3_r', 'tab10_r', 'tab20_r', 'tab20b_r', 'tab20c_r']

def create_design_matrix(x, y, degree):
    polydegs = sum(i for i in range(1, degree + 2)) # Number of combinations in polynomial
    
    X = np.ones((len(x), polydegs))
    column = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            # Create design matrix with columns 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3, ...
            input = (x**i * y**j)
            X[:,column] = input
            column += 1
    return X


# Load the terrain
terrain1 = imread('/Users/hishem/Documents/GitHub/FYS-STK4155/Project1/Datafiles/nepal_2.tif')
# Show the terrain
plt.figure(figsize=(6,6))
plt.title('Terrain over Nepal')
plt.imshow(terrain1, cmap='inferno')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.show()


print(terrain1)
print(terrain1.shape)
print(len(terrain1[0]))

x = np.linspace(0,terrain1.shape[0],terrain1.shape[0])
y = np.linspace(0,terrain1.shape[1],terrain1.shape[1])
x,y = np.meshgrid(x,y)

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

# Plot the surface.
surf = ax.plot_surface(x, y, terrain1, cmap='inferno',
                       linewidth=0, antialiased=False)

# Customize the z axis.<
ax.set_zlim(np.min(terrain1), np.max(terrain1))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


'''"""model terrain data without scaling, calculate mse and r2 score for different polynomial degrees and plot the surface"""
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


terrain1 = imread('/Users/hishem/Documents/GitHub/FYS-STK4155/Project1/Datafiles/nepal_2.tif')
z = terrain1

x = np.linspace(0,z.shape[0],z.shape[0])
y = np.linspace(0,z.shape[1],z.shape[1])
xm,ym = np.meshgrid(x,y)

x_flat = xm.flatten()
y_flat = ym.flatten()
z_flat = z.flatten()

maxdegree = 10

mse = []
r2 = []

for d in range(1, maxdegree+1):
    X = create_design_matrix(x_flat, y_flat, d)

    X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2, random_state=42)

    beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train

    z_model = X @ beta

    mse.append(mean_squared_error(z_flat, z_model))
    r2.append(r2_score(z_flat, z_model))

    print(f"Degree: {d}, MSE: {mse[-1]:.4f}, R2: {r2[-1]:.4f}")

fig = plt.figure(figsize=(8, 4))
plt.plot(range(1, maxdegree+1), mse, '-o', label='MSE')
plt.plot(range(1, maxdegree+1), r2, '-o', label='R2')
plt.xlabel('Polynomial degree')
plt.ylabel('Error')
plt.title('MSE and R2 score for terrain data - unscaled')
plt.legend()'''



from sklearn.metrics import mean_squared_error, r2_score    
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
"""Model terrain data with scaling, calculate MSE and R2 score for different polynomial degrees and plot the surface"""

terrain1 = imread('/Users/hishem/Documents/GitHub/FYS-STK4155/Project1/Datafiles/nepal_2.tif')
z = terrain1

x = np.linspace(0, z.shape[0], z.shape[0])
y = np.linspace(0, z.shape[1], z.shape[1])
xm, ym = np.meshgrid(x, y)

x_flat = xm.flatten()
y_flat = ym.flatten()
z_flat = z.flatten()

maxdegree = 2

mse_scaled = np.zeros(maxdegree)
r2_scaled = np.zeros_like(mse_scaled)

lam_ridge = 1e-4
lam_lasso = 1e-5


for d in range(1, maxdegree + 1):
    X = create_design_matrix(x_flat, y_flat, d)
    
    print('Scaling and splitting...')
    #Min max scale
    X_scaled = X.copy()
    X_scaled[:,1:] = (X[:,1:] - np.min(X[:,1:], axis=0)) / (np.max(X[:,1:], axis=0) - np.min(X[:,1:], axis=0))
    z_scaled = (z_flat - np.min(z_flat)) / (np.max(z_flat) - np.min(z_flat))

    #train test split
    X_train, X_test, z_train, z_test = train_test_split(X_scaled, z_scaled, test_size=0.2, random_state=42)

    print('Fitting models...')

    Model_ols = LinearRegression(fit_intercept=False)
    # Model_ridge = Ridge(alpha=lam_ridge)
    # Model_lasso = Lasso(alpha=lam_lasso)

    poly = PolynomialFeatures(degree=d)
    print('ols...')
    pred_ols = Model_ols.fit(poly.fit_transform(X_train),z_train).predict(poly.fit_transform(X_test))

    
    mse_scaled[d] = np.array([mean_squared_error(z_test,pred_ols)])
    r2_scaled[d] = np.array([r2_score(z_test,pred_ols)])



    print(f"Degree: {d}, MSE OLS: {mse_scaled[d]:.4f}")
    print(f"Degree: {d}, R2 OLS: {r2_scaled[d]:.4f}")

    # print('ridge...')
    # pred_ridge = Model_ridge.fit(poly.fit_transform(X_train),z_train).predict(poly.fit_transform(X_test))
    # print('lasso...')
    # pred_lasso = Model_lasso.fit(poly.fit_transform(X_train),z_train).predict(poly.fit_transform(X_test))
    # print('done')

#     mse_scaled[d] = np.array([mean_squared_error(z_test,pred_ols), mean_squared_error(z_test,pred_ridge), mean_squared_error(z_test,pred_lasso)])
#     r2_scaled[d] = np.array([r2_score(z_test,pred_ols), r2_score(z_test,pred_ridge), r2_score(z_test,pred_lasso)])

#     print(f"Degree: {d}, MSE OLS: {mse_scaled[d,0]:.4f}, MSE Ridge: {mse_scaled[d,1]:.4f}, MSE Lasso: {mse_scaled[d,2]:.4f}")
#     print(f"Degree: {d}, R2 OLS: {r2_scaled[d,0]:.4f}, R2 Ridge: {r2_scaled[d,1]:.4f}, R2 Lasso: {r2_scaled[d,2]:.4f}")

# fig = plt.figure(figsize=(8, 4))
# for i in range(3):
#     plt.plot(range(1, maxdegree + 1), mse_scaled[:,i], '-o', label=f'MSE {["OLS", "Ridge", "Lasso"][i]}')
#     plt.plot(range(1, maxdegree + 1), r2_scaled[:,i], '-o', label=f'R2 {["OLS", "Ridge", "Lasso"][i]}')
# plt.plot(range(1, maxdegree + 1), mse_scaled, '-o', label='MSE (scaled)')
# plt.plot(range(1, maxdegree + 1), r2_scaled, '-o', label='R2 (scaled)')
# plt.xlabel('Polynomial degree')
# plt.ylabel('Error')
# plt.title('MSE and R2 score for terrain data - scaled')
# plt.legend()
# plt.show()

plt.plot(range(1, maxdegree + 1), mse_scaled, '-o', label='MSE (scaled)')
plt.plot(range(1, maxdegree + 1), r2_scaled, '-o', label='R2 (scaled)')
plt.xlabel('Polynomial degree')
plt.ylabel('Error')
plt.title('MSE and R2 score for terrain data - scaled')
plt.legend()
plt.show()


