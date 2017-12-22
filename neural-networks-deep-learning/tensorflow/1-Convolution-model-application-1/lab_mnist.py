import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *
from cnn import *
from keras.datasets import mnist

#%matplotlib inline
np.random.seed(1)


# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("mnist/", one_hot=True)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plt.subplot(221)
# plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
# plt.subplot(222)
# plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
# plt.subplot(223)
# plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
# plt.subplot(224)
# plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# # show the plot
# plt.show()
indexes = 10
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train/=255
X_test/=255

# Y_train = np_utils.to_categorical(y_train, number_of_classes)
# Y_test = np_utils.to_categorical(y_test, number_of_classes)
y_train = convert_to_one_hot(y_train, indexes).T
y_test = convert_to_one_hot(y_test, indexes).T


_, _, parameters = model(X_train, y_train, X_test, y_test)