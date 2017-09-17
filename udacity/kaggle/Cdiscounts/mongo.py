import io
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

def load_preprocess_training_batch(batch_i, batch_size):
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