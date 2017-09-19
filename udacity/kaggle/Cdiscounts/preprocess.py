import base64
import pickle

import bson
import io
import numpy as np

from PIL import Image

def load(file):
    with open(file, mode='rb') as file:
        return pickle.load(file, encoding='latin1')

def save(file, data):
    with open(file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def normalize_categories():
    categories = {}
    n = 1
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
            count += 1

            print("\r Iteration %s, count %s" % (iteration, count), end='')
            im = Image.open(io.BytesIO(pic['picture']))
            arr = np.array(list(im.getdata())) / 255
            arr = np.reshape(arr, (180, 180, 3))
            #arr = np.reshape(arr, -1)


            image_data = {str(category_id): arr.tobytes()}
            append_to_bson('data/preprocesed_images_with_categories.bson', image_data)

            if count < 5000:
                append_to_bson('data/5000_train.bson', image_data)
            if count >= 5000 and count < 7000:
                append_to_bson('data/2000_valid.bson', image_data)
            if count >= 7000 and count < 9000:
                append_to_bson('data/2000_test.bson', image_data)



def append_to_bson(file, data):
    with open(file, 'ab') as handle:
        data = bson.BSON.encode(data)
        handle.write(data)

# data = bson.decode_file_iter(open('data/preprocesed_images_with_categories.bson', 'rb'))
# for iteration, row in enumerate(data):
#    print(row)
load_preprocess_training_batch()

#data = bson.decode_file_iter(open('data/MY_FIRST.bson', 'rb'))
#for iteration, row in enumerate(data):
#    print(row)