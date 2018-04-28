from cocos.batch import BatchableNode
import cocos.collision_model as cm


class ObjectProvider:
    def __init__(self, keyboard, collision, rightPanelCollision):
        self.keyboard = keyboard
        self.collision = collision
        self.rightPanelCollision = rightPanelCollision

    def checkIntersec(self, object):
        collisions = self.collision.objs_colliding(object)

        if collisions:
            return True

        return False

    def checkIntersecWithRightPanel(self, object):
        collisions = self.rightPanelCollision.objs_colliding(object)

        if collisions:
            for collision in collisions:
                return collision

        return False

    def getFakeObject(self, position, width = 2, height = 2):
        obj = BatchableNode()
        obj.cshape = cm.AARectShape(position, width // 2,height // 2)
        return obj

    def getNearObject(self, x, y, objects):
        dx = 20
        wall = self.getObjectByPoints(x - dx, y, objects)
        if wall:
            x, y = wall.position
            return (x + wall.width, y)

        wall = self.getObjectByPoints(x + dx, y, objects)
        if wall:
            x, y = wall.position
            return (x - wall.width, y)

        wall = self.getObjectByPoints(x, y + dx, objects)
        if wall:
            x, y = wall.position
            return (x, y - wall.height)

        wall = self.getObjectByPoints(x, y - dx, objects)
        if wall:
            x, y = wall.position
            return (x, y + wall.height)

    def getObjectByPoints(self, x, y, objects):
        fakeObj = self.getFakeObject((x,y))
        collisions = self.collision.objs_colliding(fakeObj)
        if collisions:
            for obj in collisions:
                return obj