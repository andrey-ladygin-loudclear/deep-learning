import random
from os import scandir, listdir

from PIL import Image
from cocos import sprite, layer
import cocos.collision_model as cm
from os.path import join

from cocos.director import director
from copy import copy

from cocos.sprite import Sprite

blockSize = 32


class Map(layer.Layer):
    def __init__(self):
        super().__init__()
        self.collisions = cm.CollisionManagerBruteForce()
        self.load_background()

    def addBrick(self, x, y, spriteBlock):
        x = x // blockSize * blockSize
        y = y // blockSize * blockSize

        spriteObj = copy(spriteBlock)
        spriteObj.position = (x, y)
        spriteObj.cshape = cm.AARectShape(spriteObj.position, spriteObj.width//2, spriteObj.height//2)

        if self.collisions.objs_colliding(spriteObj):
            return

        self.collisions.add(spriteObj)
        self.add(spriteObj, z=2)

    def removeBrick(self, x, y):
        x = x // blockSize * blockSize
        y = y // blockSize * blockSize
        point = Point(x, y)
        obj = self.collisions.any_near(point, 1)
        if obj:
            self.collisions.remove_tricky(obj)
            self.remove(obj)

    def generate_new_background(self):
        block_size = 32

        images = ['assets/background/177.jpg', 'assets/background/178.jpg', 'assets/background/179.jpg',
                  'assets/background/180.jpg', 'assets/background/181.jpg']

        x_blockCount = 140
        y_blockCount = 140

        out = Image.new('RGBA', (block_size*x_blockCount, block_size*y_blockCount))
        img = [Image.open(src) for src in images]

        for i in range(x_blockCount):
            for j in range(y_blockCount):
                x = i * 32
                y = j * 32
                out.paste(random.choice(img), (x, y))

        out.save("assets/image.png")
        out.show()

    def load_background(self):
        sprite = Sprite("assets/image.png")

class Point():
    def __init__(self, x, y):
        self.cshape = cm.AARectShape((x,y), 2, 2)