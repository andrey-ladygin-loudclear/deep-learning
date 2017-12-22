from urllib.request import urlretrieve
from os.path import isfile, isdir

from tqdm import tqdm
import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cnn_udacity_utils as utils
import cnn_udacity as cnn
import lab_cifar_different_models as models


batch_id = 1
sample_id = 5
#utils.display_stats('cifar-10-batches-py', batch_id, sample_id)

#utils.preprocess_and_save_data('cifar-10-batches-py')
valid_features, valid_labels = pickle.load(open('cifar-10-batches-py/preprocess_dev.p', mode='rb'))
print(valid_features.shape)
#https://www.coursera.org/learn/convolutional-neural-networks/lecture/A9lXL/simple-convolutional-network-example
#https://www.youtube.com/watch?v=mynJtLhhcXk

models.tests()

raise EOFError

tf.reset_default_graph()

# Inputs
x = cnn.neural_net_image_input((32, 32, 3))
y = cnn.neural_net_label_input(10)
keep_prob = cnn.neural_net_keep_prob_input()

# Model
logits = cnn.conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

epochs = 100
batch_size = 64
keep_probability = 0.5


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    cost = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
    validation_accuracy = session.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.0})
    print('Cost = {0} - Validation Accuracy = {1}'.format(cost, validation_accuracy))


# print('Checking the Training on a Single Batch...')
# with tf.Session() as sess:
#     # Initializing the variables
#     sess.run(tf.global_variables_initializer())
#
#     # Training cycle
#     for epoch in range(epochs):
#         batch_i = 1
#         for batch_features, batch_labels in utils.load_preprocess_training_batch(batch_i, batch_size):
#             sess.run(optimizer, feed_dict={x: batch_features, y: batch_labels, keep_prob: keep_probability})
#         print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
#         print_stats(sess, batch_features, batch_labels, cost, accuracy)





save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in utils.load_preprocess_training_batch(batch_i, batch_size):
                sess.run(optimizer, feed_dict={x: batch_features, y: batch_labels, keep_prob: keep_probability})
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)