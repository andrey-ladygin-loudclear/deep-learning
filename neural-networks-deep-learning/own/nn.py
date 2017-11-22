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

def tanh_deriviate(A):
    return (1 - np.power(A, 2))

# def sigmoid_deriviate(A):
#     return (1 - np.power(A, 2))

class Layer():
    W = None
    b = None
    A = None
    dW = None
    db = None
    activation_function = None
    activation_derivative_function = None

    def __init__(self, n_x, n_h, act=None, act_derivative=None):
        self.W = np.random.randn(n_h,n_x) * 0.01
        self.b = np.zeros((n_h, 1))
        self.activation_function = act
        self.activation_derivative_function = act_derivative

    def __str__(self):
        return "Layer: " + str(self.activation_function)

    def remember_activation(self, A):
        self.A = A

    def get_activated_layer(self):
        return self.A

    def set_derivatives(self, dW, db):
        self.dW = dW
        self.db = db

class NN():
    layers = []
    Y = None
    X = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_Y(self, Y):
        self.Y = Y

    def set_X(self, X):
        self.X = X

    def forward_propagation(self, X):

        # print('forward_propagation')

        A = X

        for layer in self.layers:
            W = layer.W
            b = layer.b

            Z = np.dot(W, A) + b

            #A = np.tanh(Z)
            A = layer.activation_function(Z)

            # print('W',W.shape, W)
            # print('A',A.shape, A)
            # print('b',b.shape, b)
            # print('Z',Z.shape, Z)
            # print('layer.activation_function',layer.activation_function)
            #print('W',W,'A',A,'b',b,'Z',Z,'A',A)

            layer.remember_activation(A)

        return A

    def backward_propagation(self):
        # print('backward_propagation')
        m = self.X.shape[1]
        dZ = None
        dW = None
        db = None
        W = None
        L_len = len(self.layers)

        for i in range(L_len):
            layer = self.layers[-i-1]
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
                # print('A-Y', dZ)
            else:
                #g = (1 - np.power(A, 2))
                if not layer.activation_derivative_function:
                    raise Exception('Layer <' + str(layer) + '> - has not derivative of activation function')

                g = layer.activation_derivative_function(A)
                dZ = np.multiply(np.dot(W.T, dZ),  g)

            dW = (1/m) * np.dot(dZ, A_from_next_layer.T)
            db = (1/m) * np.sum(dZ, axis=1,keepdims=True)
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
                print ("Cost after iteration %i: %f" %(i, cost))

        return cost

    def predict(self, X):
        A = self.forward_propagation(X)
        predictions = np.round(A)
        return predictions