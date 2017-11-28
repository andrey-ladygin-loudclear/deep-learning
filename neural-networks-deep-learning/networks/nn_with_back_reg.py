import matplotlib.pyplot as plt
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
        return "Layer: " + str(self.number) + \
               ', n_x:' + str(self.n_x) + \
               ', n_h:' + str(self.n_h) + \
               ', A:' + str(self.A.shape) + \
               ', dropout:' + str((self.D.shape if self.D is not None else 'None')) + \
               ', ' + str(self.activation_function)

    def init_weights(self):
        self.W = np.random.randn(self.n_h,self.n_x) / np.sqrt(self.n_x)
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

    def forward_propagation(self, X, keepprob=1):
        np.random.seed(1)
        A = X

        for layer in self.layers:
            W = layer.W
            b = layer.b

            Z = np.dot(W, A) + b

            A = layer.activation_function(Z)

            if not layer.is_last_layer:
                d = np.random.rand(A.shape[0], A.shape[1])
                d = d < keepprob
                #A = np.multiply(A, d)
                A = A * d
                A = A / keepprob
                layer.remember_dropout(d)

            layer.remember_activation(A)

        return A

    def backward_propagation(self, keepprob=1):
        np.random.seed(1)
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

                if dropout is not None:
                    dA = dA * dropout
                    dA = dA / keepprob

                dZ = np.multiply(dA,  g)

            dW = (1./m) * np.dot(dZ, A_from_next_layer.T)
            db = (1./m) * np.sum(dZ, axis=1, keepdims=True)
            W = layer.W
            layer.set_derivatives(dW,db)

    # dZ3 = A3 - Y
    # dW3 = 1./m * np.dot(dZ3, A2.T)
    # db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    # dA2 = np.dot(W3.T, dZ3)
    # ### START CODE HERE ### (≈ 2 lines of code)
    # dA2 = dA2 * D2              # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    # dA2 = dA2 / keep_prob              # Step 2: Scale the value of neurons that haven't been shut down
    # ### END CODE HERE ###
    # dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    # dW2 = 1./m * np.dot(dZ2, A1.T)
    # db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    #
    # dA1 = np.dot(W2.T, dZ2)
    # ### START CODE HERE ### (≈ 2 lines of code)
    # dA1 = dA1 * D1              # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    # dA1 = dA1 / keep_prob              # Step 2: Scale the value of neurons that haven't been shut down
    # ### END CODE HERE ###
    # dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    # dW1 = 1./m * np.dot(dZ1, X.T)
    # db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)

    def compute_cost(self):
        m = self.Y.shape[1]

        result_layer = self.layers[-1]
        A = result_layer.get_activated_layer()

        # logprobs = np.multiply(np.log(A), self.Y) + np.multiply(np.log(1 - A), (1 - self.Y))
        # cost = - (1./m) * np.nansum(logprobs)

        logprobs = np.multiply(-np.log(A),self.Y) + np.multiply(-np.log(1 - A), 1 - self.Y)
        cost = 1./m * np.nansum(logprobs)

        return cost

    def update_parameters(self, learning_rate=1.2):
        for layer in self.layers:
            layer.W = layer.W - learning_rate * layer.dW
            layer.b = layer.b - learning_rate * layer.db

    def nn_model(self, num_iterations=10000, print_cost=False, learning_rate=1.2, keepprob=1):
        np.random.seed(1)
        costs = []

        for i in range(0, num_iterations):
            self.forward_propagation(self.X, keepprob=keepprob)
            cost = self.compute_cost()

            self.backward_propagation(keepprob=keepprob)

            self.update_parameters(learning_rate=learning_rate) #0.6

            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                costs.append(cost)
                print ("Cost for LOGISTIC REGRESSION after iteration %i: %f" %(i, cost))

        # plot the cost
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        return cost

    def predict(self, X):
        A = self.forward_propagation(X)
        predictions = np.round(A)
        return predictions