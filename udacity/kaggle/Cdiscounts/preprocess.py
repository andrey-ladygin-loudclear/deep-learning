import base64
import pickle

import bson
import io
import numpy as np

from PIL import Image
from tensorflow.contrib.keras.python.keras.utils import np_utils

im_w = im_h = 180

def set_images_sdimensions(w, h):
    im_w, im_h = w, h

def load(file):
    with open(file, mode='rb') as file:
        return pickle.load(file, encoding='latin1')

def save(file, data):
    with open(file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def normalize_categories():
    categories = {}
    n = 0
    data = bson.decode_file_iter(open('data/train.bson', 'rb'))
    for iteration, row in enumerate(data):
        print("\riteration %s" % (iteration), end="")
        category_id = row['category_id']

        if category_id not in categories:
            categories[category_id] = n
            n+=1

    save('data/categories.p', categories)


def load_preprocess_training_batch():
    data = bson.decode_file_iter(open('data/train.bson', 'rb'))
    categories = load('data/categories.p')
    batch = []

    count = 0
    for iteration, row in enumerate(data):

        category_id = row['category_id']
        category_id = categories[category_id]


        for e, pic in enumerate(row['imgs']):
            append_to_bson('data/1000_test.bson', pic)
            count += 1
            if count > 2000:
                return

            # print("\r Iteration %s, count %s" % (iteration, count), end='')
            # im = Image.open(io.BytesIO(pic['picture']))
            # arr = np.array(list(im.getdata())) / 255
            # arr = np.reshape(arr, (90, 90, 3))
            # #arr = np.reshape(arr, -1)
            #
            #
            # image_data = {str(category_id): arr.tobytes()}
            # append_to_bson('data/preprocesed_images_with_categories.bson', image_data)
            #
            # if count < 5000:
            #     append_to_bson('data/5000_train.bson', image_data)
            # if count >= 5000 and count < 7000:
            #     append_to_bson('data/2000_valid.bson', image_data)
            # if count >= 7000 and count < 9000:
            #     append_to_bson('data/2000_test.bson', image_data)



def append_to_bson(file, data):
    with open(file, 'ab') as handle:
        data = bson.BSON.encode(data)
        handle.write(data)

# data = bson.decode_file_iter(open('data/preprocesed_images_with_categories.bson', 'rb'))
# for iteration, row in enumerate(data):
#    print(row)
#load_preprocess_training_batch()

#data = bson.decode_file_iter(open('data/MY_FIRST.bson', 'rb'))
#for iteration, row in enumerate(data):
#    print(row)


def load_batch_Data(start=0, end=1000, batch_number=1, batch_size=128):
    data = bson.decode_file_iter(open('data/train.bson', 'rb'))
    categories = load('data/categories.p')

    # print("batch_number %s" % (batch_number))
    packs = (end - start) // batch_size
    batch_number = max(batch_number % (packs+1), 1)
    start_from = start + (batch_number - 1) * batch_size
    end_to = start + (batch_number) * batch_size

    # print("packs %s" % (packs))
    # print("batch_number %s" % (batch_number))
    # print("start_from %s" % (start_from))
    # print("end_to %s" % (end_to))

    labels = []
    features = []

    count = 0
    for iteration, row in enumerate(data):

        category_id = row['category_id']
        category_id = categories[category_id]
        categorical = np_utils.to_categorical(category_id, len(categories))[0]

        for e, pic in enumerate(row['imgs']):

            if count >= end_to:
                labels = np.array(labels)
                features = np.array(features)
                return labels, features

            if count >= start_from:
                im = Image.open(io.BytesIO(pic['picture']))
                im.thumbnail((im_w, im_h), Image.ANTIALIAS)
                arr = np.array(list(im.getdata())) / 255
                arr = np.reshape(arr, (im_w, im_h, 3))

                labels.append(categorical)
                features.append(arr)

                #yield labels, arr

            count += 1

    labels = np.array(labels)
    features = np.array(features)
    return labels, features

#def load_train_data(batch_i, batch_size):

def load_train_data(batch_i, batch_size):
    labels, features = load_batch_Data(0, 100000, batch_i, batch_size)
    return batch_features_labels(features, labels, batch_size)


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_valid_data():
    labels, features = load_batch_Data(100000, 130000, 1, 500)
    return features, labels