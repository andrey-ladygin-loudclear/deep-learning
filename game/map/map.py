import random
from os import scandir, listdir

from PIL import Image
from cocos import sprite, layer
import cocos.collision_model as cm
from os.path import join

from cocos.director import director
from copy import copy

from cocos.sprite import Sprite

from game.map.utils import generate_image

blockSize = 32


class Map(layer.Layer):
    def __init__(self):
        super().__init__()
        self.collisions = cm.CollisionManagerBruteForce()
        self.load_background()

    def addBrick(self, x, y, spriteBlock):
        x = x // blockSize * blockSize
        y = y // blockSize * blockSize

        if isinstance(spriteBlock, str):
            spriteObj = Sprite(spriteBlock)
        else:
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
        generate_image()

    def load_background(self):
        block_size = 32
        sprite = Sprite("assets/image.png")
        #sprite.scale = 0.1
        sprite.position = (sprite.width/2, sprite.height/2)
        self.add(sprite)
        w, h = sprite.width, sprite.height
        top_wall = ['assets/objects/226.jpg','assets/objects/227.jpg']
        side_wall = ['assets/objects/220.jpg','assets/objects/221.jpg']

        for i in range(w//block_size):
            self.addBrick(i*32, 0, random.choice(top_wall))
            self.addBrick(i*32, h-block_size/2, random.choice(top_wall))

        for i in range(h//block_size):
            self.addBrick(0, i*32, random.choice(side_wall))
            self.addBrick(w-block_size/2, i*32, random.choice(side_wall))


class Point():
    def __init__(self, x, y):
        self.cshape = cm.AARectShape((x,y), 2, 2)