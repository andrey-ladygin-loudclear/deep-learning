from urllib.request import urlretrieve
from os.path import isfile, isdir

from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cnn_udacity_utils as utils


batch_id = 1
sample_id = 5
#utils.display_stats('cifar-10-batches-py', batch_id, sample_id)

# utils.preprocess_and_save_data('cifar-10-batches-py')
valid_features, valid_labels = pickle.load(open('cifar-10-batches-py/preprocess_dev.p', mode='rb'))
print(len(valid_features))
#https://www.youtube.com/watch?v=mynJtLhhcXk