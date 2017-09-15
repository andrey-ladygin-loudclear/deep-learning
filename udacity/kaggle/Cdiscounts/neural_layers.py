import tensorflow as tf

def neural_net_image_input(image_shape):
    n_input_1 = image_shape[0]
    n_input_2 = image_shape[1]
    n_input_3 = image_shape[2]
    return tf.placeholder(tf.float32,[None, n_input_1, n_input_2, n_input_3], name='x')


def neural_net_label_input(n_classes):
    return tf.placeholder(tf.float32, [None, n_classes], name='y')


def neural_net_keep_prob_input():
    return tf.placeholder(tf.float32, None, name='keep_prob')


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    depth = x_tensor.get_shape().as_list()[-1]
    bias = tf.Variable(tf.zeros(conv_num_outputs))

    c_strides = [1, conv_strides[0], conv_strides[1], 1]
    p_ksize = [1, pool_ksize[0], pool_ksize[1], 1]
    p_strides = [1, pool_strides[0], pool_strides[1], 1]

    # 2x2x5x10
    weight= tf.Variable(tf.truncated_normal([conv_ksize[0], conv_ksize[1], depth, conv_num_outputs]))


    conv = tf.nn.conv2d(x_tensor, weight, c_strides, 'SAME') + bias
    conv = tf.nn.relu(conv)

    pool = tf.nn.max_pool(conv, p_ksize, p_strides, 'SAME')

    return pool

def flatten(x_tensor):
    b, w, h, d = x_tensor.get_shape().as_list()
    img_size = w * h * d
    return tf.reshape(x_tensor, [-1, img_size])

def fully_conn(x_tensor, num_outputs):
    shape = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.truncated_normal([shape[-1], num_outputs]))
    bias = tf.Variable(tf.zeros(num_outputs))
    return tf.nn.relu(tf.add(tf.matmul(x_tensor, weight), bias))

def output(x_tensor, num_outputs):
    shape = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.truncated_normal([shape[-1], num_outputs], stddev=0.1))
    bias = tf.Variable(tf.zeros(num_outputs))
    return tf.add(tf.matmul(x_tensor, weight), bias)
