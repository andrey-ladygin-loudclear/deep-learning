'''
A linear regression learning algorithm example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.00001
training_epochs = 2000
display_step = 50

# Training Data
train_X = numpy.asarray([1,2,3,4,5,6,7,8,9,10])
train_Y = numpy.asarray([1,4,9,8,25,12,49,16,91,20])
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
W2 = tf.Variable(rng.randn(), name="weight2")
W3 = tf.Variable(rng.randn(), name="weight3")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
X2 = tf.multiply(X, X)
X3 = tf.multiply(X2, X)
X4 = tf.multiply(X3, X)
# pred = tf.add(tf.multiply(X4, W), b)

pred = X*X*X*W + b + W2*X*X + W3*X

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                  "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    #plt.plot(train_X, sess.run(W) * train_X*train_X + sess.run(b), label='Fitted line')


    plt.plot(train_X,
             train_X*train_X*train_X*sess.run(W) + sess.run(b) + sess.run(W2)*train_X*train_X + sess.run(W3)*train_X,
             label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([1,2,3,4,5,6,7,8,9,10])
    test_Y = numpy.asarray([1,4,9,8,25,12,49,16,91,20])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    # plt.plot(test_X, test_Y, 'bo', label='Testing data')
    # plt.plot(train_X, sess.run(W) * train_X*train_X + sess.run(b), label='Fitted line')
    # plt.legend()
    # plt.show()