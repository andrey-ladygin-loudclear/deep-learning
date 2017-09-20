import bson
import base64
from PIL import Image
import io

#bson_file = open('data/test.bson', 'rb')
#b = bson.loads(bson_file.read())
#print(len(b))
import mongo
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


w, h = 180, 180

categories = preprocess.load('data/categories.p')
num_of_categories = len(categories)


epochs = 100
batch_size = 256
keep_probability = 0.5

convNetwork = ConvolutionNetwork()
convNetwork.build(w, h, num_of_categories)
convNetwork.train(epochs, batch_size, keep_probability)
