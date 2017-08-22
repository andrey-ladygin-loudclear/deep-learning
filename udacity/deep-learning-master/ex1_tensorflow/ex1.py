import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)


#conver types
tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))   # 1

# Quiz Solution
# Note: You can't run code in this tab
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x,y),tf.cast(tf.constant(1), tf.float64))

# TODO: Print z from a session
with tf.Session() as sess:
    output = sess.run(z)
    print(output)


# BIAS
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))

#WEIGHTS
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

# MULTIPLAY AND ADD BIAS
tf.add(tf.matmul(input, w), b)

# SOFTMAX
x = tf.nn.softmax([2.0, 1.0, 0.2])


# !_@!#*!_(@#*(!@*#)!@*#)!@*#
output = None
logit_data = [2.0, 1.0, 0.1]
logits = tf.placeholder(tf.float32)
# TODO: Calculate the softmax of the logits
softmax = tf.nn.softmax(logit_data)
with tf.Session() as sess:
    # TODO: Feed in the logit data
    output = sess.run(softmax)
# BUT IT CAN BE LIKE THIS
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)
    softmax = tf.nn.softmax(logits)
    with tf.Session() as sess:
        output = sess.run(softmax, feed_dict={logits: logit_data})
