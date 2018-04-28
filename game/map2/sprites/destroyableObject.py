from cocos import sprite
import cocos.collision_model as cm


class destroyableObject(sprite.Sprite):
    def __init__(self, name, position = (0,0)):
        super(destroyableObject, self).__init__(name)

        self.type = 'destroyable'
        self.src = str(name)
        self.position = position

    def _update_position(self):
        super(destroyableObject, self)._update_position()

        self.cshape = cm.AARectShape(
            self.position,
            self.width // 2,
            self.height // 2
        )