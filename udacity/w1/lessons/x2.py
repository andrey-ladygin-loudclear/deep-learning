import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist

n = 28

X = tf.placeholder(tf.float32, [None, n, n, 1])# 1 - greyscaling
W = tf.Variable(tf.zeros([n*n, 10]))
b = tf.Variable(tf.zeros([10]))

init = tf.initialize_all_variables()


Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, n*n]), W) + b)
'''
tf.reshape(X, [-1, n*n]) - reshape 28x28 to a one-dimensional vector 784
'''

Y_ = tf.placeholder(tf.float32, [None, 10]) # one-hot encoded, WHAT IT IS?

#loss (error) function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

#% of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


optimizer = tf.train.GradientDescentOptimizer(0.003)# learning rate
train_step = optimizer.minimize(cross_entropy)


sess = tf.Session()
sess.run(init)

for i in range(1000):
    # Load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X:  batch_X, Y_:batch_Y}

    #train
    sess.run(train_step, feed_dict=train_data)

    # success
    a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

    # seccess on test data?
    test_data={X: mnist.test.images, Y_: mnist.test.labes}
    a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)