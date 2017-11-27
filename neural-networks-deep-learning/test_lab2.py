import numpy as np
import matplotlib.pyplot as plt
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model
import scipy.io
from networks import nn_with_back_reg as nn
from networks import functions as nf

def load_planar_dataset(seed):

    np.random.seed(seed)

    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y

def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);

    return train_X, train_Y, test_X, test_Y

train_X, train_Y, test_X, test_Y = load_2D_dataset()

# layers_dims = [X.shape[0], 20, 3, 1]

Network = nn.NN()
Network.set_X(train_X)
Network.set_Y(train_Y)

L1 = nn.Layer(train_X.shape[0], 20, act=nf.relu, act_derivative=nf.relu_deriviate)
L2 = nn.Layer(20, 3, act=nf.relu, act_derivative=nf.relu_deriviate)
L3 = nn.Layer(3, train_Y.shape[0], act=nf.sigmoid, is_last_layer=True)

Network.add_layer(L1)
Network.add_layer(L2)
Network.add_layer(L3)

Network.nn_model(num_iterations=30000, print_cost=True, learning_rate=0.7)

# parameters = model(train_X, train_Y)
# print ("On the training set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)
predictions = Network.predict(train_X, train_Y)
print(predictions)
predictions = Network.predict(test_X, test_Y)
print(predictions)