import numpy as np
from keras.datasets import imdb

np.random.seed(42)

max_features = 5000

(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)

#https://www.youtube.com/watch?v=7Tx_cewjhGQ&list=PLtPJ9lKvJ4oiz9aaL_xcZd-x0qd8G0VN_&index=13

maxlen = 80 # 80 words

X_train = sequence.pad_sequence(X_train, maxlen=maxlen)
X_test = sequence.pad_sequence(X_test, maxlen=maxlen)

model = Sequential()

# vectorial word layer
model.add(Embedding(max_features, 32, dropout=0.2))

#lstm layer
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
#dropout_W - input bindings
#dropout_U - recurent bindings

#classifing layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=64,
          np_epoch=7,
          validation_data=(X_test, y_test),
          verbose=1)

scores = model.evaluate(X_test, y_test, batch_size=64)
print('Accuracy: %.2f%%' % (scores[1]*100))