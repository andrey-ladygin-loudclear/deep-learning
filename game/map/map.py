from os import scandir, listdir

from cocos import sprite, layer
import cocos.collision_model as cm
from os.path import join

from cocos.director import director
from copy import copy

blockSize = 32


class Map(layer.Layer):
    def __init__(self):
        super().__init__()
        self.collisions = cm.CollisionManagerBruteForce()

    def addBrick(self, x, y, spriteBlock):
        x = x // blockSize * blockSize
        y = y // blockSize * blockSize

        spriteObj = copy(spriteBlock)
        spriteObj.position = (x, y)
        spriteObj.cshape = cm.AARectShape(spriteObj.position, spriteObj.width//2, spriteObj.height//2)

        if self.collisions.objs_colliding(spriteObj):
            return

        self.collisions.add(spriteObj)
        self.add(spriteObj)

    def removeBrick(self, x, y):
        x = x // blockSize * blockSize
        y = y // blockSize * blockSize
        point = Point(x, y)
        obj = self.collisions.any_near(point, 1)
        if obj:
            self.collisions.remove_tricky(obj)
            self.remove(obj)


class Point():
    def __init__(self, x, y):
        self.cshape = cm.AARectShape((x,y), 2, 2)