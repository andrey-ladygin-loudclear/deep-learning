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

#%matplotlib inline
np.random.seed(1)



def show_image(index, X):
    image = Image.fromarray(X[index], 'RGB')
    plt.imshow(image)
    plt.show()

def show_rand_images(X):
    fig, axes1 = plt.subplots(5,5,figsize=(3,3))
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(X)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(X[i:i+1][0])
    plt.show()

def get_data():
    datadict = unpickle('cifar-10-batches-py/data_batch_1')
    X = datadict[b"data"]
    #X = np.ndarray(shape=(0,3072))
    Y = datadict[b'labels']

    datadict = unpickle('cifar-10-batches-py/data_batch_2')
    X = np.insert(X, 0, datadict[b"data"], axis=0)
    Y = datadict[b'labels'] + Y

    datadict = unpickle('cifar-10-batches-py/data_batch_3')
    X = np.insert(X, 0, datadict[b"data"], axis=0)
    Y = datadict[b'labels'] + Y

    datadict = unpickle('cifar-10-batches-py/data_batch_4')
    X = np.insert(X, 0, datadict[b"data"], axis=0)
    Y = datadict[b'labels'] + Y

    datadict = unpickle('cifar-10-batches-py/data_batch_5')
    X = np.insert(X, 0, datadict[b"data"], axis=0)
    Y = datadict[b'labels'] + Y

    X = X.reshape(X.shape[0], 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    Y = np.array(Y)
    return X, Y

def get_test_data():
    datadict = unpickle('cifar-10-batches-py/test_batch')
    # for i in datadict: print(i)
    X = datadict[b"data"]
    Y = datadict[b'labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    Y = np.array(Y)
    return X, Y

X_train, Y_train = get_data()
X_test, Y_test = get_test_data()
indexes = len(np.unique(Y_test))
# print("X", X.shape)
# print("Y", )
# print("test_X", test_X.shape)
# print("test_Y", test_Y.shape)
#
# raise EnvironmentError
#
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
#
# index = 6
# plt.imshow(X_train_orig[index])
# print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

X_train = X_train/255.
X_test = X_test/255.
Y_train = convert_to_one_hot(Y_train, indexes).T
Y_test = convert_to_one_hot(Y_test, indexes).T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}


# _, _, parameters = model(X_train, Y_train, X_test, Y_test, minibatch_size=256, learning_rate=0.5)
_, _, parameters = model(X_train, Y_train, X_test, Y_test)

fname = "images/thumbs_up.jpg"
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64))
plt.imshow(my_image)