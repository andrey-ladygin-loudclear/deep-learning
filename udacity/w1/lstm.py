import numpy as np
from keras.datasets import imdb

np.random.seed(42)

max_features = 5000

(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)

#https://www.youtube.com/watch?v=7Tx_cewjhGQ&list=PLtPJ9lKvJ4oiz9aaL_xcZd-x0qd8G0VN_&index=13