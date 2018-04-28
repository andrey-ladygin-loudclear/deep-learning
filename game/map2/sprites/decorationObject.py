from cocos import sprite
import cocos.collision_model as cm


class decorationObject(sprite.Sprite):
    def __init__(self, name, position = (0,0)):
        super(decorationObject, self).__init__(name)

        self.type = 'decoration'
        self.src = name
        self.position = position

    def _update_position(self):
        super(decorationObject, self)._update_position()

        self.cshape = cm.AARectShape(
            self.position,
            self.width // 2,
            self.height // 2
        )