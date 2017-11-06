from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from PIL import Image
import json

#X_train = np.array(np.zeros(500))
#Y_train = np.array(np.zeros(500))
X_train = []
Y_train = []

X_test = []
Y_test = []

with open('1.json') as data_file:
    data = json.load(data_file)

img = Image.open("1.png")

for i in range(50):
    row = i // 12
    col = i % 12
    w,h = 150, 150
    area = (col*w, row, (i+1)*w, h)
    cropped_img = img.crop(area)
    cropped_img.thumbnail((75, 75), Image.ANTIALIAS)
    matrix = np.array(cropped_img)
    reshaped = matrix.reshape(-1)
    #data[i]['image'] = matrix

    weights = []
    for el in data[i]:
        weights.append(data[i][el])

    X_train.append(reshaped)
    Y_train.append(weights)

    if(len(X_test) < 5):
        X_test.append(reshaped)
        Y_test.append(weights)


# boston = learn.datasets.load_dataset('boston')
# x, y = boston.data, boston.target
# X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
#     x, y, test_size=0.2, random_state=42)

# for but in data:
#     X_train.
#
# print(x, y)

print('Features setted!')

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

total_len = X_train.shape[0]
weights_len = len(Y_train[0])

# Parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 10
display_step = 1
dropout_rate = 0.9
# Network Parameters
n_hidden_1 = 75 # 1st layer number of features
n_hidden_2 = 200 # 2nd layer number of features
n_hidden_3 = 200
n_hidden_4 = 256

print(X_train.shape)

n_input = 22500#X_train.shape[1]
n_classes = 1

# tf Graph input
x = tf.placeholder("float", [None, 22500])
y = tf.placeholder("float", [None, weights_len])

print('Params setted!')

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, 0.1))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1)),
    'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1)),
    'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(pred-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(total_len/batch_size)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = Y_train[i*batch_size:(i+1)*batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            print(batch_x, batch_y)
            _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        # sample prediction
        # label_value = batch_y
        # estimate = p
        # err = label_value-estimate
        # print ("num batch:", total_batch)
        #
        # # Display logs per epoch step
        # if epoch % display_step == 0:
        #     print ("Epoch:", '%04d' % (epoch+1), "cost=", \
        #            "{:.9f}".format(avg_cost))
        #     print ("[*]----------------------------")
        #     for i in xrange(3):
        #         print ("label value:", label_value[i], \
        #                "estimated value:", estimate[i])
        #     print ("[*]============================")

    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))