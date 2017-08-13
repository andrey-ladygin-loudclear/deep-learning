
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)
rides.head()
rides[:24*10].plot(x='dteday', y='cnt')

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()

""" scaling """
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std
""" scaling """


''' split data to training set'''
# Save data for approximately the last 21 days
test_data = data[-21*24:]

# Now remove the test data from the data set
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
''' split data to training set'''

# Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.

    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            #### Implement the forward pass here ####
            ### Forward pass ###

            #a1 = X # input values

            #z2 = np.dot(self.weights_input_to_hidden.T, a1)
            #a2 = self.activation_function(z2) # input to hidden

            #z3 = np.dot(self.weights_hidden_to_output.T, a2)
            #a3 = self.activation_function(z3) # hidden to output

            #delta3 = a3 - y

            #a2_deriv = a2 * (1 - a2)
            #delta2 = np.dot(self.weights_hidden_to_output, delta3) * a2_deriv

            #delta_weights_i_h += delta2 * (a1[:, None])
            #delta_weights_h_o += delta3 * (a2[:, None])

            hidden_input = np.dot(X, self.weights_input_to_hidden)
            hidden_output = self.activation_function(hidden_input)

            final_inputs = np.dot(hidden_output, self.weights_hidden_to_output)
            final_outputs = self.activation_function(final_inputs)

            error = y - final_inputs

            hidden_error = np.dot(self.weights_hidden_to_output, error)
            hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

            output_error = np.dot(self.weights_input_to_hidden, hidden_error)
            output_error_term = final_outputs * (1 - final_outputs)

            delta_weights_h_o += error * hidden_output[:, None]
            delta_weights_i_h += hidden_error_term * X[:, None]

            #hidden_inputs = np.dot(X, self.weights_input_to_hidden) #  (56) X (56,5) -> (1,5) shaped output
            #hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer ->(1,5)
            #final_inputs =  np.dot(hidden_outputs, self.weights_hidden_to_output) # (1,5)X(5,10) -> (1,10) shape
            #final_outputs = final_inputs # (1,10) shape
            #error = y - final_outputs # (1,10)
            #hidden_error = np.dot(error, self.weights_hidden_to_output.T) # (1,10)X(10,5) -> (1,5)
            #The hidden_error gets the dimension right and it makes the computation run without breaking but error calculation #here is wrong, additionally you are calculating additional weights for output layers


        # TODO: Update the weights - Replace these values with your calculations.
        self.weights_hidden_to_output += self.lr * (delta_weights_h_o / n_records) # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * (delta_weights_i_h / n_records) # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''

        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = (final_inputs) # signals from final output layer

        return final_outputs

def MSE(y, Y):
    return np.mean((y-Y)**2)



import sys

### Set the hyperparameters here ###
iterations = 500
learning_rate = 0.3
hidden_nodes = 2
output_nodes = 1

X = np.array([
    [1],
    [2],
    [3],
])
y = np.array([10, 20, 30])

N_i = X.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

for i in range(iterations):
    network.train(X, y)


test = np.array([
    [100]
])

res = network.run(test)

print 'res: '+str(res)

print network.weights_input_to_hidden
print network.weights_hidden_to_output

'''
N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']

    network.train(X, y)

    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()
'''