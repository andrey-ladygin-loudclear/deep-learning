import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # TODO: Implement Function
    n_input_1 = image_shape[0]
    n_input_2 = image_shape[1]
    n_input_3 = image_shape[2]
    return tf.placeholder(tf.float32,[None, n_input_1, n_input_2, n_input_3], name='x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, [None, n_classes], name='y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, None, name='keep_prob')


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

    #layer = conv2d_maxpool(x, 16, (4,4), (1,1), (2,2), (2,2))
    layer = create_convolution_layers(x)
    tf.nn.dropout(layer, keep_prob=keep_prob)

    #layer = flatten(layer)
    layer = tf.contrib.layers.flatten(layer)
    #layer = fully_conn(layer, 400)
    layer = tf.contrib.layers.fully_connected(layer, 400)
    layer = tf.nn.dropout(layer, keep_prob)

    #res = output(layer,10)
    res = tf.contrib.layers.fully_connected(layer, 10, activation_fn=None)

    return res


def create_convolution_layers(X):
    Z1 = create_conv2d(X, 10, strides=[2,2], w_name='W1')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    #Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')
    Z2 = create_conv2d(P1, 20, strides=[2,2], w_name='W2')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    Z3 = create_conv2d(P2, 40, strides=[2,2], w_name='W3')
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    return P3

def create_conv2d(X, conv_num_outputs, strides, w_name):
    depth = X.get_shape().as_list()[-1]
    w_size = [strides[0], strides[1], depth, conv_num_outputs]
    c_strides = [1, strides[0], strides[1], 1]
    W = tf.get_variable(w_name, w_size, initializer=tf.contrib.layers.xavier_initializer(seed=0))
    Z = tf.nn.conv2d(X, W, strides=c_strides, padding='SAME')
    return Z

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """

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
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    b, w, h, d = x_tensor.get_shape().as_list()
    img_size = w * h * d
    return tf.reshape(x_tensor, [-1, img_size])

def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    shape = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.truncated_normal([shape[-1], num_outputs]))
    bias = tf.Variable(tf.zeros(num_outputs))
    return tf.nn.relu(tf.add(tf.matmul(x_tensor, weight), bias))

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    shape = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.truncated_normal([shape[-1], num_outputs], stddev=0.1))
    bias = tf.Variable(tf.zeros(num_outputs))
    return tf.add(tf.matmul(x_tensor, weight), bias)

def get_bias(n):
    return tf.Variable(tf.zeros(n))





def conv_net_check1(x, keep_prob):
    layer = create_conv2d(x, 10, strides=[2,2], w_name='W1')
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    layer = create_conv2d(layer, 20, strides=[2,2], w_name='W2')
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    layer = create_conv2d(layer, 40, strides=[2,2], w_name='W3')
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    tf.nn.dropout(layer, keep_prob=keep_prob)
    layer = tf.contrib.layers.flatten(layer)
    layer = tf.contrib.layers.fully_connected(layer, 400)
    layer = tf.nn.dropout(layer, keep_prob)
    return tf.contrib.layers.fully_connected(layer, 10, activation_fn=None)

def conv_net_check2(x, keep_prob):
    layer = create_conv2d(x, 10, strides=[2,2], w_name='W1')
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    layer = create_conv2d(layer, 20, strides=[2,2], w_name='W2')
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    layer = create_conv2d(layer, 40, strides=[2,2], w_name='W3')
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    tf.nn.dropout(layer, keep_prob=keep_prob)
    layer = tf.contrib.layers.flatten(layer)
    layer = tf.contrib.layers.fully_connected(layer, 400)
    layer = tf.nn.dropout(layer, keep_prob)
    layer = tf.contrib.layers.fully_connected(layer, 200)
    layer = tf.nn.dropout(layer, keep_prob)
    return tf.contrib.layers.fully_connected(layer, 10, activation_fn=None)

def conv_net_check3(x, keep_prob):
    layer = create_conv2d(x, 10, strides=[2,2], w_name='W1')
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    layer = create_conv2d(layer, 20, strides=[2,2], w_name='W2')
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    layer = create_conv2d(layer, 40, strides=[2,2], w_name='W3')
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    layer = create_conv2d(layer, 80, strides=[2,2], w_name='W4')
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    tf.nn.dropout(layer, keep_prob=keep_prob)
    layer = tf.contrib.layers.flatten(layer)
    layer = tf.contrib.layers.fully_connected(layer, 400)
    layer = tf.nn.dropout(layer, keep_prob)
    layer = tf.contrib.layers.fully_connected(layer, 200)
    layer = tf.nn.dropout(layer, keep_prob)
    return tf.contrib.layers.fully_connected(layer, 10, activation_fn=None)

def conv_net_check4(x, keep_prob):
    layer = create_conv2d(x, 10, strides=[2,2], w_name='W1')
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    layer = create_conv2d(layer, 20, strides=[2,2], w_name='W2')
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    layer = create_conv2d(layer, 40, strides=[2,2], w_name='W3')
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    layer = create_conv2d(layer, 80, strides=[2,2], w_name='W4')
    layer = tf.nn.relu(layer)
    layer = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    tf.nn.dropout(layer, keep_prob=keep_prob)
    layer = tf.contrib.layers.flatten(layer)
    layer = tf.contrib.layers.fully_connected(layer, 400)
    layer = tf.nn.dropout(layer, keep_prob)
    layer = tf.contrib.layers.fully_connected(layer, 200)
    layer = tf.nn.dropout(layer, keep_prob)
    layer = tf.contrib.layers.fully_connected(layer, 100)
    layer = tf.nn.dropout(layer, keep_prob)
    return tf.contrib.layers.fully_connected(layer, 10, activation_fn=None)