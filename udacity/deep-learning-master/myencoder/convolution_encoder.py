import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', validation_size=0)

img = mnist.train.images[2]

#plt.imshow(img.reshape(28, 28), cmap='Greys_r')
#plt.show()

learning_rate = 0.001
#image_size = mnist.train.images.shape[1]

inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')


#reshape image
img_4d = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
#get shape as list
img_4d.get_shape().as_list()
#https://www.youtube.com/watch?v=ETdaP_bBNWc

### Encoder
conv1 = tf.layers.conv2d(inputs_, 16, (3,3), padding='same', activation=tf.nn.relu)
# Now 28x28x16

maxpool1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
# Now 14x14x16

conv2 = tf.layers.conv2d(maxpool1, 8, (3,3), padding='same', activation=tf.nn.relu)
# Now 14x14x8

maxpool2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
# Now 7x7x8

conv3 = tf.layers.conv2d(maxpool2, 8, (3,3), padding='same', activation=tf.nn.relu)
# Now 7x7x8

encoded = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
# Now 4x4x8

### Decoder
upsample1 = tf.image.resize_nearest_neighbor(encoded, (7,7))
# Now 7x7x8
conv4 = tf.layers.conv2d(upsample1, 8, (3,3), padding='same', activation=tf.nn.relu)
# Now 7x7x8
upsample2 = tf.image.resize_nearest_neighbor(conv4, (14,14))
# Now 14x14x8
conv5 = tf.layers.conv2d(upsample2, 8, (3,3), padding='same', activation=tf.nn.relu)
# Now 14x14x8
upsample3 = tf.image.resize_nearest_neighbor(conv5, (28,28))
# Now 28x28x8
conv6 = tf.layers.conv2d(upsample3, 16, (3,3), padding='same', activation=tf.nn.relu)
# Now 28x28x16

logits = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)
#Now 28x28x1

decoded = tf.nn.sigmoid(logits, name='decoded')

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(0.001).minimize(cost)



# depth = x_tensor.get_shape().as_list()[-1]
# bias = tf.Variable(tf.zeros(conv_num_outputs))
#
# c_strides = [1, conv_strides[0], conv_strides[1], 1]
# p_ksize = [1, pool_ksize[0], pool_ksize[1], 1]
# p_strides = [1, pool_strides[0], pool_strides[1], 1]
#
# # 2x2x5x10
# weight= tf.Variable(tf.truncated_normal([conv_ksize[0], conv_ksize[1], depth, conv_num_outputs]))
#
#
# conv = tf.nn.conv2d(x_tensor, weight, c_strides, 'SAME') + bias
# conv = tf.nn.relu(conv)
#
# pool = tf.nn.max_pool(conv, p_ksize, p_strides, 'SAME')
