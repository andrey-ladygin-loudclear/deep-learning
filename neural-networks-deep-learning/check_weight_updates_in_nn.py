import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from own.dnn_app_utils_v2 import load_data

np.random.seed(1)
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

print(train_x_orig.shape)