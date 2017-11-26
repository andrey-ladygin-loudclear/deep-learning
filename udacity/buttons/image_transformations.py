from PIL import Image
import numpy as np
import json
from pprint import pprint

with open('1.json') as data_file:
    data = json.load(data_file)

#1854, 6484, 12 pictures per row

img = Image.open("1.png")

for i in range(500):
    row = i // 12
    col = i % 12
    w,h = 150, 150
    area = (col*w, row, (i+1)*w, h)
    cropped_img = img.crop(area)
    cropped_img.thumbnail((75, 75), Image.ANTIALIAS)
    #cropped_img.show()
    matrix = np.array(cropped_img)
    reshaped = matrix.reshape(-1)
    #print(reshaped.shape)
    #print(reshaped.reshape([75, 75, 4]).shape)
    data[i]['image'] = matrix

    #if i > 15: break