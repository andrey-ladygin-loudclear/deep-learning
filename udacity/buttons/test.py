from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from PIL import Image
import json

#X_train = np.array(np.zeros(500))
#Y_train = np.array(np.zeros(500))
X_train = np.array([1,2,3,4,5,6,7,8,9,10])
Y_train = np.array([1,4,6,8,10,12,14,16,18,20])

X_test = np.array([11,12,14,16])
Y_test = np.array([22,24,28,32])


X_train = X_train[:, None]
Y_train = Y_train[:, None]
X_test = X_test[:, None]
Y_test = Y_test[:, None]

# Parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 2
display_step = 1
dropout_rate = 0.9


# Network Parameters
n_hidden_1 = 1 # 1st layer number of features
n_hidden_2 = 200 # 2nd layer number of features
n_hidden_3 = 200
n_hidden_4 = 1

n_input = 1#X_train.shape[1]
n_classes = 1

total_len = X_train.shape[0]
weights_len = 1

# tf Graph input
x = tf.placeholder("float", [None, 1])
y = tf.placeholder("float", [None, weights_len])

print('Params setted!')

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, 0.1))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
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
            #print(batch_x, batch_y)
            _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

            # sample prediction
            label_value = batch_y
            estimate = p
            err = label_value - estimate
            print ("num batch:", total_batch)

            # Display logs per epoch step
            # if epoch % display_step == 0:
            #     print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            #     print ("[*]----------------------------")
            #     for i in range(2):
            #         print ("label value:", label_value[i], "estimated value:", estimate[i])
            #     print ("[*]============================")

    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # print(X_test, X_test[:, None].shape)
    # print(Y_test, Y_test.shape)

    #for i in range(len(X_test)):
    #    print ("Accuracy:", accuracy.eval({x: X_test[i], y: Y_test[i]}))
    print ("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))

    feed_dict = {x: np.array([[50]])}
    # print(accuracy)
    # print(accuracy.eval({x:t[:, None]}))
    print('estimate',pred)
    classification = sess.run(cost, feed_dict)
    print(classification)