import tensorflow as tf

def normalize(x):
    return x/255

def one_hot_encode(labels, num_classes):
    return tf.contrib.layers.one_hot_encoding(labels, num_classes)