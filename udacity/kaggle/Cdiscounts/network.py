import tensorflow as tf

import neural_layers as nl
import mongo

def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)

    layer = nl.conv2d_maxpool(x, 16, (4,4), (1,1), (2,2), (2,2))
    tf.nn.dropout(layer, keep_prob=keep_prob)

    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    layer = nl.flatten(layer)


    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)

    layer = nl.fully_conn(layer,400)
    layer = tf.nn.dropout(layer, keep_prob)


    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    res = nl.output(layer,10)


    # TODO: return output
    return res

class ConvolutionNetwork:
    cost = None
    optimizer = None
    accuracy = None

    x = None
    y = None
    keep_prob = None

    #valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
    valid_features, valid_labels = mongo.get_valid_features_and_labels()

    def build(self, w, h, num_of_categories):
        # Remove previous weights, bias, inputs, etc..
        tf.reset_default_graph()

        # Inputs
        x = nl.neural_net_image_input((w, h, 3))
        y = nl.neural_net_label_input(num_of_categories)
        keep_prob = nl.neural_net_keep_prob_input()

        # Model
        logits = conv_net(x, keep_prob)

        # Name logits Tensor, so that is can be loaded from disk after training
        logits = tf.identity(logits, name='logits')

        # Loss and Optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    def train(self, epochs, batch_size, keep_probability):
        save_model_path = './image_classification'

        print('Training...')
        with tf.Session() as sess:
            # Initializing the variables
            sess.run(tf.global_variables_initializer())

            # Training cycle
            for epoch in range(epochs):
                # Loop over all batches
                print("Epoch %s" % (epoch))
                n_batches = 5
                for batch_i in range(1, n_batches + 1):
                    print("Batch i %s" % (batch_i))
                    for batch_features, batch_labels in mongo.load_preprocess_training_batch(batch_i, batch_size):
                        self.train_neural_network(sess, self.optimizer, keep_probability, batch_features, batch_labels)
                    print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                    self.print_stats(sess, batch_features, batch_labels, self.cost, self.accuracy)

            # Save Model
            saver = tf.train.Saver()
            save_path = saver.save(sess, save_model_path)

    def train_neural_network(self, session, optimizer, keep_probability, feature_batch, label_batch):
        session.run(optimizer, feed_dict={self.x: feature_batch, self.y: label_batch, self.keep_prob: keep_probability})

    def print_stats(self, session, feature_batch, label_batch, cost, accuracy):
        cost = session.run(cost, feed_dict={self.x: feature_batch, self.y: label_batch, self.keep_prob: 1.0})
        validation_accuracy = session.run(accuracy, feed_dict={self.x: self.valid_features, self.y: self.valid_labels, self.keep_prob: 1.0})
        print('Cost = {0} - Validation Accuracy = {1}'.format(cost, validation_accuracy))
