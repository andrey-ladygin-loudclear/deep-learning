import json
import random

from PIL import Image

src1 = '../assets/177.jpg'
src2 = '../assets/178.jpg'
src3 = '../assets/179.jpg'

x_blockCount = 140
y_blockCount = 140

mw = 32 * x_blockCount
mh = 32 * y_blockCount

out = Image.new('RGBA', (mw, mh))
img = [
    Image.open(src1),
    Image.open(src2),
]

for i in range(x_blockCount):
    for j in range(y_blockCount):
        x = i * 32
        y = j * 32
        out.paste(random.choice(img), (x, y))

out.save("image.png")
out.show()