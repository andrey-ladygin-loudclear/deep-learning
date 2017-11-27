import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def tanh_deriviate(X):
    return (1 - np.power(X, 2))

def sigmoid_deriviate(X):
    return sigmoid(X) * (1 - sigmoid(X))

def relu(X):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    return np.max(X, 0)

def relu_deriviate(X):
    return X < 0