import numpy as np


class Layer():
    W = None # weights
    b = None # bias
    A = None # Activation
    D = None # dropout
    dW = None # weights derivative
    db = None # bias derivative
    n_x = None # size features layer
    n_h = None # size params layer
    number = 0 # number of layer
    is_last_layer = False
    activation_function = None
    activation_derivative_function = None

    def __init__(self, n_x, n_h, act=None, act_derivative=None, is_last_layer=False):
        self.n_x = n_x
        self.n_h = n_h
        self.activation_function = act
        self.activation_derivative_function = act_derivative
        self.is_last_layer = is_last_layer

        self.init_weights()

    def __str__(self):
        return "Layer: " + str(self.number) + ', n_x:' + str(self.n_x) + ', n_h:' + str(self.n_h) + ', ' + str(self.activation_function)

    def init_weights(self):
        self.W = np.random.randn(self.n_h,self.n_x) * 0.01
        self.b = np.zeros((self.n_h, 1))

    def get_shape(self):
        return self.n_x, self.n_h

    def set_number(self, number):
        self.number = number

    def remember_activation(self, A):
        self.A = A

    def remember_dropout(self, D):
        self.D = D

    def get_activated_layer(self):
        return self.A

    def set_derivatives(self, dW, db):
        self.dW = dW
        self.db = db

class NN():
    layers = []
    Y = None
    X = None
    layer_number = 0

    def add_layer(self, layer):
        self.layer_number += 1
        layer.set_number(self.layer_number)
        self.layers.append(layer)

    def set_Y(self, Y):
        self.Y = Y

    def set_X(self, X):
        self.X = X

    def forward_propagation(self, X, keepprob = 1):
        A = X

        for layer in self.layers:
            W = layer.W
            b = layer.b

            if not layer.is_last_layer:
                d = np.random.rand(A.shape[0], A.shape[1]) < keepprob
                A = np.multiply(A, d)
                A /= keepprob
                layer.remember_dropout(d)

            Z = np.dot(W, A) + b

            A = layer.activation_function(Z)

            layer.remember_activation(A)

        return A

    def backward_propagation(self, keepprob = 1):
        # print('backward_propagation')
        m = self.X.shape[1]
        dZ = None
        dW = None
        db = None
        W = None
        L_len = len(self.layers)

        for i in range(L_len):
            layer = self.layers[-i-1]

            # shoulb be updated
            #https://www.coursera.org/learn/deep-neural-network/lecture/C9iQO/vanishing-exploding-gradients
            A = layer.get_activated_layer()

            # if L_len-i != L_len:
            #     prev_layer = self.layers[-i]
            #     A_from_prev_layer = prev_layer.get_activated_layer()
            #     g = prev_layer.activation_derivative_function(A_from_prev_layer)

            try:
                next_layer = self.layers[-i-2]
                A_from_next_layer = next_layer.get_activated_layer()
            except IndexError:
                A_from_next_layer = self.X

            if dZ is None:
                dZ = A - self.Y
            else:
                #g = (1 - np.power(A, 2))
                if not layer.activation_derivative_function:
                    raise Exception('Layer <' + str(layer) + '> - has not derivative of activation function')

                # seems for g(z)=z derivative should be not activated layer
                # if z=w*x => a=g(z)=z then derivative should be w (z'=w => g(w))
                g = layer.activation_derivative_function(A)
                dropout = layer.D

                dA = np.dot(W.T, dZ)

                if dropout:
                    dA = dA * dropout
                    dA = dA / keepprob

                dZ = np.multiply(dA,  g)

            dW = (1/m) * np.dot(dZ, A_from_next_layer.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            W = layer.W
            layer.set_derivatives(dW,db)

    def compute_cost(self):
        m = self.Y.shape[1]

        result_layer = self.layers[-1]
        A = result_layer.get_activated_layer()

        logprobs = np.multiply(np.log(A), self.Y) + np.multiply(np.log(1 - A), (1 - self.Y))
        cost = - (1/m) * np.sum(logprobs)

        return cost

    def update_parameters(self, learning_rate=1.2):
        for layer in self.layers:
            layer.W = layer.W - learning_rate * layer.dW
            layer.b = layer.b - learning_rate * layer.db

    def nn_model(self, num_iterations=10000, print_cost=False, learning_rate=1.2):
        np.random.seed(3)

        for i in range(0, num_iterations):
            self.forward_propagation(self.X)
            cost = self.compute_cost()

            self.backward_propagation()

            self.update_parameters(learning_rate=learning_rate) #0.6

            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print ("Cost for LOGISTIC REGRESSION after iteration %i: %f" %(i, cost))

        return cost

    def predict(self, X):
        A = self.forward_propagation(X)
        predictions = np.round(A)
        return predictions