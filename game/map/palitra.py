from os import scandir, listdir

from cocos import sprite, layer
import cocos.collision_model as cm
from os.path import join

from cocos.director import director


class Palitra(layer.Layer):
    iconSizeK = 0.5

    def __init__(self):
        super().__init__()
        self.collisions = cm.CollisionManagerBruteForce()
        self.load()

    def load(self):
        for folder in ['background', 'objects', 'undestroyable_objects', 'unmovable_background']:
            path = 'assets/' + folder
            imgs = sorted(listdir(path))
            for src in imgs:
                self.addImg("{}/{}".format(path, src), folder)

    def updatePosition(self, width, height, viewPoint):
        x, y = viewPoint
        self.position = -x, -y

    def resizeMap(self, width, height, viewPoint):
        start_col = 0

        objs = [
            [land for land in self.collisions.objs if isinstance(land, BackgroundSprite)],
            [land for land in self.collisions.objs if isinstance(land, ObjectSprite)],
            [land for land in self.collisions.objs if isinstance(land, UndestroyableObject)],
            [land for land in self.collisions.objs if isinstance(land, UnmovableBackground)]
        ]

        self.updatePosition(width, height, viewPoint)
        max_cols_per_background_objects = 7
        offset = 2

        for items in objs:
            row = 0
            col = 0
            for land in sorted(items, key=lambda x: x.src):
                x = land.width * self.iconSizeK + land.width * (start_col + col) + offset
                y = land.height * self.iconSizeK + land.height * row + offset
                col += 1
                if col > max_cols_per_background_objects:
                    row += 1
                    col = 0
                land.position = (x, y)
                land.cshape = cm.AARectShape(land.position, land.width//2, land.height//2)

            start_col += max_cols_per_background_objects + 4


    def addImg(self, src, type):
        land = None

        if type == 'background': land = BackgroundSprite(src)
        if type == 'objects': land = ObjectSprite(src)
        if type == 'undestroyable_objects': land = UndestroyableObject(src)
        if type == 'unmovable_background': land = UnmovableBackground(src)
        assert land is not None, 'Land is none! type is undefined!'
        land.cshape = cm.AARectShape(land.position, land.width//2, land.height//2)
        land.src = src
        self.add(land)
        self.collisions.add(land)

    def click(self, x, y):
        point = Point(x, y)
        return self.collisions.any_near(point, 1)


class Point():
    def __init__(self, x, y):
        self.cshape = cm.AARectShape((x,y), 2, 2)


class BackgroundSprite(sprite.Sprite):
    def __copy__(self):
        return BackgroundSprite(self.image, self.position, rotation=self.rotation, scale=self.scale,
                                opacity=self.opacity, color=self.color, anchor=self.anchor)


class ObjectSprite(sprite.Sprite):
    def __copy__(self):
        return ObjectSprite(self.image, self.position, rotation=self.rotation, scale=self.scale,
                            opacity=self.opacity, color=self.color, anchor=self.anchor)


class UndestroyableObject(sprite.Sprite):
    def __copy__(self):
        return UndestroyableObject(self.image, self.position, rotation=self.rotation, scale=self.scale,
                                   opacity=self.opacity, color=self.color, anchor=self.anchor)


class UnmovableBackground(sprite.Sprite):
    def __copy__(self):
        return UnmovableBackground(self.image, self.position, rotation=self.rotation, scale=self.scale,
                                   opacity=self.opacity, color=self.color, anchor=self.anchor)
