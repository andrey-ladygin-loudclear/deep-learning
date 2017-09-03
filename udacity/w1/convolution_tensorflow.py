import numpy

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Convolution2D, MaxPooling2D, Dropout, Conv2D
#from tensorflow.contrib.keras.python.keras.utils import np_utils
from keras.utils import np_utils
import tensorflow as tf

numpy.random.seed(42)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#normalize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#convert metki classov to categories
#[7] -> [0 0 0 0 0 0 1 0 0]
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)




# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)





#print(X_train.shape)
#print(X_train[0])

# layers as queue
model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3), activation='relu'))
#32 - map of attributes
# 3x3 its core of convolution
#input_shape=(32, 32, 3) = 3 its  RGB, 32x32 img dimension
model.add(Conv2D(32, (3, 3), activation='relu'))

#reduce dimension
model.add(MaxPooling2D(pool_size=(2,2)))
#from square 2x2 we select max value: 2x2 -> 1x1

#regularization
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


#classificator
model.add(Flatten()) # convert from 2 dimension to one dimension

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

#output layer
model.add(Dense(10, activation='softmax'))


#compile
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=32,
          epochs=25,
          validation_split=0.1,
          shuffle=True)

scores = model.evaluate(X_test, Y_test, verbose=0)

print('Accuracy: %.2f%%' % scores[1]*100)