import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

class Layer():
    W = None
    b = None
    activation_function = None

    def __init__(self, n_x, n_h, act = None):
        self.W = np.random.randn(n_h,n_x) * 0.01
        self.b = np.zeros((n_h, 1))
        self.activation_function = act

class NN():
    layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_propagation(self, X, parameters):

        A = X

        for layer in self.layers:
            W = layer.W
            b = layer.b

            Z = np.dot(W, A) + b

            A = np.tanh(Z)

        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        ### END CODE HERE ###

        # Implement Forward Propagation to calculate A2 (probabilities)
        ### START CODE HERE ### (≈ 4 lines of code)
        Z1 = np.dot(W1,X)+b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2,A1)+b2
        A2 = sigmoid(Z2)
        ### END CODE HERE ###

        assert(A2.shape == (1, X.shape[1]))

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}

        return A2, cache