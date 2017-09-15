import bson
import base64
from PIL import Image
import io
import network

bson_file = open('data/train_example.bson', 'rb')
b = bson.loads(bson_file.read())

img_data = b['imgs'][0]
img_data = img_data['picture']

#with open("imageToSave.png", "wb") as fh:
    #fh.write(base64.decodebytes(img_data))
#    fh.write(img_data)

image = Image.open(io.BytesIO(img_data))
image.show()

w, h = image.size


epochs = 100
batch_size = 128
keep_probability = 0.5


network.build()