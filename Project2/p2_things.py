import scipy.special
import autograd.numpy as np
from sklearn.metrics import mean_squared_error,  r2_score


# functions from project 1
def design_with_in(x1f, x2f, degreeplus1):
    n = x1f.size
    num_of_monoms = scipy.special.binom(degreeplus1 + 1, degreeplus1 -1)
    X = np.zeros((n, int(num_of_monoms)))
    c = 0
    for i in range(degreeplus1):
        for j in range(degreeplus1 - i):
            X[:,c] = x1f**i * x2f**j
            #print('{} {}'.format(i,j))
            #print(X[:,c])
            #print('x_1^{} * x_2^{}'.format(i,j)) 
            c += 1
    return X

''' generate design matrix without intercept column for flattend x1, x2 '''
def design_no_in(x1f, x2f, degreeplus1):
    n = x1f.size
    # 1. setting up design matrix
    # we work here with intercept column
    # the designmatrix s.t. we have the features 1, x1, x2, x1x2, x1^2, x2^2, ....
    # with max totaldegree 5
    # number of possible monoms up to degree 5 for 2 variables is 2+5 choose 5
    num_of_monoms = scipy.special.binom(degreeplus1 + 1, degreeplus1 -1)
    X = np.zeros((n, int(num_of_monoms))) # afterwards the intercept column is removed
    c = 0
    for i in range(degreeplus1):
        for j in range(degreeplus1 - i):
            X[:,c] = x1f**i * x2f**j
            #print('{} {}'.format(i,j))
            #print(X[:,c])
            #print('x_1^{} * x_2^{}'.format(i,j)) 
            c += 1
    # remove now intercept column
    X = X[:,1:]
    return X



## PROJECT SPECIFIC THINGS
# Defining some activation functions
def ReLU(z):
    return np.where(z > 0, z, 0)

# Derivative of the ReLU function
def ReLU_der(z):
    return np.where(z > 0, 1, 0)

def leaky_ReLU(z):
    return np.where(z > 0, z, 0.1*z)

def leaky_ReLU_der(z):
    return np.where(z > 0, 1, 0.1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_der(z):
    return sigmoid(z)*(1-sigmoid(z))


def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]


def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict))

def cost_batch(layers, input, activation_funcs, target): # changed wrt mse and not mse_batch
    predict = feed_forward_batch(input, layers, activation_funcs)
    return mse(predict, target)
    
def mse(predict, target):
    return np.mean((predict - target) ** 2)

def mse_der(predict, target):
    return 2/len(predict)*(predict - target)

# FFNN for batched
def create_layers_FFNN(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.normal(loc=0, scale=1, size=(layer_output_size, i_size))
        b = np.zeros(layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size

    #print('LAYERS')
    #print([(layer[0].shape, layer[1].shape) for layer in layers])
    return layers

def feed_forward_batch(a, layers, activation_funcs):
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a @ W.T + b.T
        a = activation_func(z)
        #print('shape for a: {}'.format(a.shape))
    return a # due to labsession guy: z

def feed_forward_saver_FFNN_batch(a, layers, activation_funcs):
    layer_inputs = []
    zs = []
    #print('\na beginning feedforward is {}'.format(a))
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = a @ W.T + b.T
        a = activation_func(z)

        zs.append(z)

    #print('\na from feedforward is {}'.format(a))
    #layer_inputs.append(a) due to guy from labsession... but i don't think so
    return layer_inputs, zs, a #but labsaession guy says z

def Id(x): return x

def one(x): return 1

def backpropagation_FFNN_batch(
    ainput, layers, target, activation_funcs, activation_ders, cost_der=mse_der
):
       
    layer_inputs, zs, predict = feed_forward_saver_FFNN_batch(ainput, layers, activation_funcs)
    num_obs = target.shape[0]
    layer_grads = [() for _ in range(len(layers) + 1)]

    # We loop over the layers, from the last to the first
    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]
        #print('i = {}'.format(i))

        if i == len(layers)-1:
            #print('predict and target {}'.format((predict,target)))
            # For last layer we use cost derivative as dC_da(L) can be computed directly
            dC_da = cost_der(predict, target)
            #print('dC_da begin: {}'.format(dC_da))
        else:
            # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
            (W, b) = layers[i+1]
            dC_da = dC_dz @ W
            
        dC_dz = dC_da * activation_der(z)
        #print('dC_dz: {}'.format(dC_dz))
        # now dC_dW aand dC_db are lists with the matrices  for the observations
        #dC_dW = [np.transpose(dC_dz[i] * layer_input[i].reshape(-1,1)) for i in range(num_obs)] # old but NO IDEA
        dC_dW = np.dot(layer_input.T, dC_dz).T / num_obs
        
        #dC_db = [dC_dz[i] * np.ones(z.shape[1]) for i in range(num_obs)]
        #print(dC_dW.shape)
        dC_db = np.mean(dC_dz, axis=0)
        #print(dC_db.shape)
        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads

# for stochastic gradient descent
''' OLD: which  I GUESS is incorrect, but somehow worked in lectures with this ''' 
def sum_c_i(X_sub, beta, y_sub,n, method='OLS', lamb=0, activation_funcs=[], activation_ders=[], cost_der=0): # Xsub is a X from which rows are deleted, same for y_sub
    if method == 'OLS':
        return X_sub.T @ ((X_sub @ beta) - y_sub) *2/n
    if method == 'RIDGE':
        return X_sub.T @ ((X_sub @ beta) - y_sub) *2/n  + 2*lamb*y_sub.size/n
    if method == 'FFNN':
        #print('\ny_sub = target = {}'.format(y_sub))
        return backpropagation_FFNN_batch(X_sub, beta, y_sub, activation_funcs, activation_ders, cost_der)


# stochastic gradient descent (with momentum) for OLS, NN or Ridge.
# using verbose gives accuracy or mse&r2 score, giving test data → also does this for test
# if (for ols and ridge) scaled data is used→ give scaler_y, and then y should already be scaled (same for X) but the y_test should not be scaled
def sgd_momentum(X,y, size_s, sum_c_i, method='OLS', activation_funcs=[], activation_ders=[], cost_der=mse_der,scaler_y=0, lamb=0, n_epochs=50, M=5, gamma=0.001, momentum = 0.9, verbose=False, method2='regression',X_test=0, y_test=0):
    #print(activation_funcs)
    global xk
    global change
    if method not in ['OLS', 'RIDGE','FFNN']:
        raise ValueError('no valid method chosen.')
    n = y.size
    if M > n:
        raise ValueError("can't take Minibatch of size %i from sest of size %i" %(M,n))

    verbose_esteps = 10
    m = int(y.shape[0]/M) #number of minibatches


    def xkplus1_OLSRDIGE(gradient):
        global change
        global xk
        
        change = gamma*gradient + change*momentum
        xk = xk - change
    def xkplus1_FFNN(gradient):
        global change
        global xk
        for i, ((W, b), (W_g, b_g), (changeW, changeb)) in enumerate(zip(xk, gradient, change)):
            changeW = gamma*W_g + momentum*changeW
            changeb = gamma*b_g + momentum*changeb
            W -= changeW
            b -= changeb
            xk[i] = (W, b)
            change[i] = (changeW, changeb)
    xkplus1 = None
    
    if method in  ['OLS', 'RIDGE']:
        if scaler_y == 0:
            y_true = y
        else: 
            y_true = scaler_y.inverse_transform(y)
        deg = size_s # X.shape[1]
        xk = np.random.randn(deg,1) # starting point for beta
        change = 0.0

        xkplus1 = xkplus1_OLSRDIGE
        

    if method == 'FFNN':
        xk = create_layers_FFNN(size_s[0], size_s[1]) # = layers
        change = [(np.zeros(W.shape), np.zeros(b.shape)) for (W,b) in xk]
        
        xkplus1 = xkplus1_FFNN

    
    if method2 == 'regression':
        if isinstance(X_test, np.ndarray):
            mse_r2 = np.zeros((n_epochs// verbose_esteps, 5))
        else: 
            mse_r2 = np.zeros((n_epochs// verbose_esteps, 3))
    elif method2 == 'class':
        if isinstance(X_test, np.ndarray):
            mse_r2 = np.zeros((n_epochs// verbose_esteps, 3))
        else:
            mse_r2 = np.zeros((n_epochs// verbose_esteps, 2))

    else:
        mse_r2 = np.zeros((n_epochs// verbose_esteps, 3))
    
    for e in range(n_epochs):
        shuff = np.random.choice(X.shape[0], X.shape[0], False)
        X_shuff = X[shuff]
        y_shuff = y[shuff]

        for i in range(m):
            random_index = M*np.random.randint(m) 
            Xi = X_shuff[random_index:random_index+M]
            yi = y_shuff[random_index:random_index+M]
            gradient = sum_c_i(Xi, xk, yi,n,method,lamb,activation_funcs, activation_ders,cost_der)
            xkplus1(gradient)
            
        if verbose:
            if e % verbose_esteps == verbose_esteps-1:
                if method2 == 'regression':
                    if method == 'FFNN':
                        pred = feed_forward_batch(X,xk,activation_funcs)
                        if isinstance(X_test, np.ndarray):
                            pred_test = feed_forward_batch(X_test,xk,activation_funcs)
                            mse_r2[e // verbose_esteps] = np.array([e, mean_squared_error(y,pred), r2_score(y,pred),  mean_squared_error(y_test,pred_test), r2_score(y_test,pred_test)])
                        else:
                            mse_r2[e // verbose_esteps] = np.array([e, mean_squared_error(y,pred), r2_score(y,pred)])

                    elif method == 'OLS' or method == 'RIDGE':
                        pred =  scaler_y.inverse_transform(X @ xk).reshape(-1, 1)
                        if isinstance(X_test, np.ndarray):
                            pred_test = scaler_y.inverse_transform(X_test @ xk).reshape(-1, 1)
                            mse_r2[e // verbose_esteps] = np.array([e, mean_squared_error(y_true,pred), r2_score(y_true,pred),  mean_squared_error(y_test,pred_test), r2_score(y_test,pred_test)])
                        else:
                            mse_r2[e // verbose_esteps] = np.array([e, mean_squared_error(y_true,pred), r2_score(y_true,pred)])

                elif method2 == 'class':
                    if method == 'FFNN':
                        pred = feed_forward_batch(X,xk,activation_funcs)
                        #print('prediction : {}'.format(pred.T))#pred[15:22,:].T))
                        if isinstance(X_test, np.ndarray):
    
                            pred_test = feed_forward_batch(X_test,xk,activation_funcs)
                            mse_r2[e // verbose_esteps] = np.array([e, accuracy(pred,y), accuracy(pred_test,y_test)])
                        else:
                            mse_r2[e // verbose_esteps] = np.array([e, accuracy(pred,y)])



    if verbose:
        return xk, mse_r2
    return xk
