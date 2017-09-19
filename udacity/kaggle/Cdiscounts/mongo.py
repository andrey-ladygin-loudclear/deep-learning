import io

import bson
from PIL import Image
from numpy.random.mtrand import rand
from pymongo import MongoClient
import pprint

client = MongoClient('localhost', 27017)
collection = client.cdiscount.categories_test

#res = collection.limit( 10 ).find()
# res = collection.find().skip( 2 ).limit(5)
#
# for i in res:
#     print(i['_id'])

# 10
# 21
# 27
# 32
# 67

w, h = 64, 64

def get_image_from_bytes(bytes):
    return Image.open(io.BytesIO(bytes))

def get_image(bytes, size=(64, 64)):
    img = get_image_from_bytes(bytes)
    img.thumbnail(size, Image.ANTIALIAS)
    return img

def get_one():
    return collection.find_one()

def load_preprocess_training_batch2(batch_i, batch_size):
    skip = int(rand() * collection.count())
    skip = min(skip, collection.count() - batch_size)
    res = collection.find().skip( skip ).limit( batch_size )
    batch_features = []
    batch_labels = []
    print("batch_size %s, skip %s" % (batch_size, skip))

    for item in res:
        print("Cat Id: %s, # of images is %s" % (item['_id'], len(item['imgs'])))
        for img in item['imgs']:
            print("yield")
            yield item['_id'], get_image(img['picture'], (w,h))
    #         batch_labels.append(item['_id'])
    #         batch_features.append(get_image(img['picture'], (w,h)))
    #
    # print("NUMBER: batch_i is %s" % (batch_i))
    # print("labels:")
    # print(batch_labels)
    # print("Number Of features: %s" % (len(batch_features)))
    # print("")
    #
    # return batch_features, batch_labels


import tensorflow as tf

def load_preprocess_training_batch(batch_i, batch_size):
    data = bson.decode_file_iter(open('data/train.bson', 'rb'))

    valid_features = []
    valid_labels = []

    for iteration, row in enumerate(data):

        #print("Iteration {0}, batch_i {1}, batch_size {2}".format(iteration, batch_i, batch_size))

        if iteration >= batch_i * batch_size + batch_size - 1:
            break

        if iteration >= batch_i * batch_size:
            product_id = row['_id']
            category_id = row['category_id']

            for e, pic in enumerate(row['imgs']):
                valid_labels.append(category_id)
                valid_features.append(get_image(pic['picture'], (w,h)))


                image_reader = tf.WholeFileReader()
                #img = get_image(pic['picture'], (w,h))
                img = io.BytesIO(pic['picture'])
                #img = image_reader.read(pic['picture'])

                #yield category_id, tf.image.decode_jpeg(img)
                print(tf.compat.as_bytes(pic['picture']))
                yield category_id, tf.compat.as_bytes(pic['picture'])

    return
    #return valid_features, valid_labels

def get_valid_features_and_labels(limit = None):
    data = bson.decode_file_iter(open('data/train.bson', 'rb'))

    valid_features = []
    valid_labels = []

    for iteration, row in enumerate(data):

        if limit and iteration > limit: break

        product_id = row['_id']
        category_id = row['category_id']

        for e, pic in enumerate(row['imgs']):
            valid_labels.append(category_id)
            valid_features.append(get_image(pic['picture'], (w,h)))

    return valid_features, valid_labels

#pip install pymongo
def get_valid_features_and_labels_TEST():
    data = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))
    prod_to_category = dict()

    for c, d in enumerate(data):
        product_id = d['_id']
        category_id = d['category_id'] # This won't be in Test data
        prod_to_category[product_id] = category_id
        for e, pic in enumerate(d['imgs']):
            picture = imread(io.BytesIO(pic['picture']))
            # do something with the picture, etc

    prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
    prod_to_category.index.name = '_id'
    prod_to_category.rename(columns={0: 'category_id'}, inplace=True)

    return valid_features, valid_labels

# first = collection.find_one()
#
# pprint.pprint(collection.count())
# pprint.pprint(first['_id'])
#
# for im in first['imgs']:
#     Image.open(io.BytesIO(im['picture'])).show()
#
# input("PRESS ENTER TO CONTINUE.")

# for post in posts.find():
#     ...   pprint.pprint(post)
#tags = db.mycoll.find({"category": "movie"}).distinct("tags")



# cat = get_one()
# imgs = [get_image(im['picture'], (w,h)) for im in cat['imgs']]
# print(imgs)
# print(imgs[1].size)
# print(imgs[1].show())