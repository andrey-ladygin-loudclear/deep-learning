from cocos import sprite
import cocos.collision_model as cm


class backgroundObject(sprite.Sprite):
    def __init__(self, name, position = (0,0)):
        super(backgroundObject, self).__init__(name)

        self.type = 'background'
        self.src = name
        self.position = position

    def _update_position(self):
        super(backgroundObject, self)._update_position()

        self.cshape = cm.AARectShape(
            self.position,
            self.width // 2,
            self.height // 2
        )