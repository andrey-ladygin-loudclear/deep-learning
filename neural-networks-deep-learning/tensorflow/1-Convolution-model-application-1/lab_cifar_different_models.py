from urllib.request import urlretrieve
from os.path import isfile, isdir

import pickle
from tqdm import tqdm
import tensorflow as tf
import cnn_udacity as cnn
import cnn_udacity_utils as utils

valid_features, valid_labels = pickle.load(open('cifar-10-batches-py/preprocess_dev.p', mode='rb'))

def tests():
    test(1, 'conv_net_check1')
    test(2, 'conv_net_check2')
    test(3, 'conv_net_check3')
    test(4, 'conv_net_check4')

def test(n, name):
    tf.reset_default_graph()

    # Inputs
    x = cnn.neural_net_image_input((32, 32, 3))
    y = cnn.neural_net_label_input(10)
    keep_prob = cnn.neural_net_keep_prob_input()

    # Model
    if n == 1:
        logits = cnn.conv_net_check1(x, keep_prob)
    if n == 2:
        logits = cnn.conv_net_check2(x, keep_prob)
    if n == 3:
        logits = cnn.conv_net_check3(x, keep_prob)
    if n == 4:
        logits = cnn.conv_net_check4(x, keep_prob)


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
        cost = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
        validation_accuracy = session.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.0})
        print('('+name+'), Cost = {0} - Validation Accuracy = {1}'.format(cost, validation_accuracy))

    print('Checking the Training ('+name+') on a Single Batch...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())

        # Training cycle
        for epoch in range(epochs):
            batch_i = 1
            for batch_features, batch_labels in utils.load_preprocess_training_batch(batch_i, batch_size):
                sess.run(optimizer, feed_dict={x: batch_features, y: batch_labels, keep_prob: keep_probability})
            print('Epoch {:>2}, ('+name+') CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
