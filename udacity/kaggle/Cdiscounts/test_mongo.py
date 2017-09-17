import io
import os

from PIL import Image
from numpy.random.mtrand import rand
from pymongo import MongoClient
import pprint

client = MongoClient('localhost', 27017)
collection = client.cdiscount.categories_train

total_count = collection.count()

print('Total %s' % (total_count))

#cats = 5270

# records = collection.find().limit(40)
# pprint.pprint("is %s" % (len(list(records))))

def get_all(distinct = None):
    maxLimit = 100000
    skip = 0

    while True:
        result = collection.find().skip(skip).limit(maxLimit)
        setlen = len(list(result))

        print('%s of %s, set length %s' % ((maxLimit+skip), total_count, setlen))

        for row in result:
            yield row

        if setlen < maxLimit:
            break

        skip += maxLimit


def get_all_cats():
    result = collection.find().distinct("category_id")
    setlen = len(list(result))
    return setlen


def save_images_to_filesystem():
    #os.mkdir('img')

    result = collection.find().limit(600)
    for row in result:

        try:
            os.mkdir(os.path.join('img', str(row['category_id'])))
        except FileExistsError:
            pass

        name = 0
        for im in row['imgs']:
            name+=1
            image = Image.open(io.BytesIO(im['picture']))
            image.save(os.path.join('img', str(row['category_id']), str(name) + '.jpg'))

save_images_to_filesystem()

#unique = []
#for row in get_all():
#    if row['category_id'] not in unique:
#        unique.append(row['category_id'])
#
#pprint.pprint("Total Unique Cats Count is %s" % (len(unique)))

def save_some_images_with_cats():
    result = collection.find().limit(10000)


def select(distinct = None, limit = None):
    query = collection.find()#.distinct("category_id")
    total_count = collection.count()
    maxLimit = 1000
    skip = 0
    #.skip( skip ).limit( batch_size )

    if distinct:
        query = query.distinct(distinct)

    while maxLimit + skip < total_count:
        pass

    def get(distinct = None, limit = None, skip = None):
        query = collection.find()
        if distinct: query = query.distinct(distinct)
        return query.skip(skip).limit(limit)

    cats = []
    cats_unique = []
    iteration = 0
    pprint.pprint("Total Unique Cats Count is %s" % (len(list(records))))
    return


    for row in records:
        iteration+=1
        #print("\rIteration %s of %s, num cats %s" % (iteration, total_count, len(cats)), end='')
        #cats[row['category_id']] = 1
        #if row['category_id'] not in cats:
        #    cats.append(row['category_id'])
        #cats_unique.append(row['category_id'])

    #pprint.pprint("Total Unique Cats Count is %s, distinct cats are %s" % (len(cats), len(cats_unique)))

    pprint.pprint("Total Unique Cats Count is %s" % (iteration))

def checkSomeFirstRows():
    records = collection.find().limit(20)

    for row in records:
        #pprint.pprint(collection.count())
        pprint.pprint("ID: %s" % (row['_id']))
        pprint.pprint("category ID: %s" % (row['category_id']))
        pprint.pprint("imgs : %s" % (len(row['imgs'])))
        pprint.pprint("")

        if row['category_id'] == 1000010653:
            for im in row['imgs']:
               Image.open(io.BytesIO(im['picture'])).show()



#checkSomeFirstRows()
#getUniqueCats()
#input("PRESS ENTER TO CONTINUE.")

import pandas as pd
from pymongo import MongoClient


def _connect_mongo(host, port, username, password, db):
    """ A util for making a connection to mongo """

    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)


    return conn[db]


def read_mongo(db, collection, query={}, host='localhost', port=27017, username=None, password=None, no_id=True):
    """ Read from Mongo and Store into DataFrame """

    # Connect to MongoDB
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)

    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df