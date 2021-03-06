import bson
import base64
from PIL import Image
import io

#bson_file = open('data/test.bson', 'rb')
#b = bson.loads(bson_file.read())
#print(len(b))
#import mongo
import preprocess
from network import ConvolutionNetwork


# img_data = b['imgs'][0]
# img_data = img_data['picture']
#
# #with open("imageToSave.png", "wb") as fh:
#     #fh.write(base64.decodebytes(img_data))
# #    fh.write(img_data)
#
# image = Image.open(io.BytesIO(img_data))
# image.show()
#
# w, h = image.size
#
#

w, h = 90, 90

preprocess.set_images_sdimensions(w, h)

categories = preprocess.load('data/categories.p')
num_of_categories = len(categories)


epochs = 2000
batch_size = 128
n_batches = 15
keep_probability = 0.5

valid_batch_size = 512

convNetwork = ConvolutionNetwork()
convNetwork.build(w, h, num_of_categories)
convNetwork.train(epochs, batch_size, n_batches, keep_probability)
