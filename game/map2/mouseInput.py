import os
from time import sleep

import cocos
from cocos import sprite
from cocos.layer import Layer, director, ScrollableLayer, pyglet
from cocos.text import Label
import cocos.collision_model as cm

from pyglet.window import key
from scandir import scandir

from ObjectProvider import ObjectProvider
from ButtonsProvider import ButtonsProvider


class MouseInput(ScrollableLayer):
    anchor = (0, 0)
    is_event_handler = True
    walls = []
    labels = []
    collision = cm.CollisionManagerBruteForce()
    palitraCollision = cm.CollisionManagerBruteForce()

    focusX = 1500
    focusY = 500

    currentType = 1
    currentSprite = None

    appendMode = 1

    buttonsTextHelp = "q-pallet, 1 - background, 2 - unmovable background, 3 - indestructible object, 4 - object, t - increase type"


    viewPoint = (0, 0)
    currentWidth = 0
    currentHeight = 0

    def __init__(self, keyboard, scroller):
        super(MouseInput, self).__init__()
        self.set_view(0, 0, 2000, 800, 0,0 )

        self.keyboard = keyboard
        self.scroller = scroller
        self.buttonsProvider = ButtonsProvider()
        self.objectProvider = ObjectProvider(self.keyboard, self.collision, self.palitraCollision)

        self.helperLayer = cocos.layer.Layer()
        self.buttonsInfo = Label(self.buttonsTextHelp, font_name='Helvetica', font_size=12, anchor_x='left',  anchor_y='top')
        self.text = Label("Some text", font_name='Helvetica', font_size=12, anchor_x='left',  anchor_y='bottom')
        self.helperLayer.add(self.text)
        self.helperLayer.add(self.buttonsInfo)
        self.add(self.helperLayer, z=5)

        self.palitra = cocos.layer.Layer()
        self.palitraObject = []
        self.add(self.palitra, z=2)
        self.loadPalitra()
        self.resizeMap()

        map = self.buttonsProvider.getMap()
        if map: self.loadMap(map)

    def loadMap(self, map):
        for block in map:
            try:
                x, y = block['position']
                spriteObj = sprite.Sprite(block['src'])
                spriteObj.src = block['src']
                spriteObj.position = (x, y)
                spriteObj.type = block['type']
                spriteObj.cshape = cm.AARectShape(spriteObj.position,spriteObj.width//2,spriteObj.height//2)

                self.walls.append(spriteObj)
                self.add(spriteObj)

                if spriteObj.type:
                    self.collision.add(spriteObj)
            except pyglet.resource.ResourceNotFoundException:
                pass

        # spriteObj = sprite.Sprite('backgrounds/fill.png')
        # spriteObj.src = 'backgrounds/fill.png'
        # spriteObj.position = (0, 0)
        # spriteObj.type = 0
        #
        # self.add(spriteObj)
        #
        # return

    def loadPalitra(self):
        #imgs = sorted(glob.glob('assets/*'))
        imgs = sorted(scandir('assets'))

        names = []
        for file in imgs: names.append(file.name)


        for src in sorted(names):
            src = 'assets/' + str(src)

            land = sprite.Sprite(src)
            land.cshape = cm.AARectShape(land.position, land.width//2, land.height//2)
            land.src = src
            self.palitra.add(land)
            self.palitraObject.append(land)
            self.palitraCollision.add(land)

    def resizeMap(self):
        col = 0
        row = 0
        offset = 2
        for land in self.palitraObject:
            x = land.width * .5 + land.width * col + offset - self.currentWidth // 2
            y = land.height * .5 + land.height * row + offset - self.currentHeight // 2
            col += 1
            if col > 19:
                row += 1
                col = 0
            land.position = (x, y)
            land.cshape = cm.AARectShape(land.position,land.width//2,land.height//2)

    def checkButtons(self, dt):
        x_direction = self.keyboard[key.LEFT] - self.keyboard[key.RIGHT]
        y_direction = self.keyboard[key.DOWN] - self.keyboard[key.UP]
        x, y = self.position

        if self.keyboard[key.Q]:
            self.palitra.visible = 1 - self.palitra.visible
            sleep(0.1)

        if x_direction:
            x += x_direction * 20

        if y_direction:
            y += y_direction * 20

        if x_direction or y_direction:
            self.set_view(0, 0, self.currentWidth, self.currentHeight, x, y)

        if self.keyboard[key.SPACE]:
            self.set_view(0, 0, self.currentWidth, self.currentHeight, 0, 0)

        self.setLayersPosition()

        if self.keyboard[key.S]:
            self.buttonsProvider.exportToFile(self.walls)

        if self.keyboard[key.T]:
            self.currentType += 1
            if self.currentType > 4: self.currentType = 1
            self.updateInfo('T')
            sleep(0.1)

        if self.keyboard[key.NUM_1]:
            self.currentType = 1
            self.updateInfo('1')

        if self.keyboard[key.NUM_2]:
            self.currentType = 2
            self.updateInfo('2')

        if self.keyboard[key.NUM_3]:
            self.currentType = 3
            self.updateInfo('3')

        if self.keyboard[key.NUM_4]:
            self.currentType = 4
            self.updateInfo('4')

    def setLayersPosition(self):
        x, y = self.getCoordByViewPoint(0, 0)
        self.helperLayer.position = (x, y)
        self.palitra.position = (x + 2, y + 16)
        #self.text.position = (x, y)
        #self.buttonsInfo.position = (x, y + self.currentHeight)


    def resize(self, width, height):
        self.viewPoint = (width // 2, height // 2)
        self.currentWidth = width
        self.currentHeight = height

        self.buttonsInfo.position = (0, self.currentHeight)
        self.setLayersPosition()
        #self.resizeMap()

    def on_mouse_motion(self, x, y, dx, dy):
        pass
        #self.addBrick(x, y)

    def on_mouse_move(self, x, y):
        print 'mouse move'

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        x, y = self.getCoordByViewPoint(x, y)
        leftClick = buttons == 1
        rightClick = buttons == 4

        if not self.palitra.visible:
            if leftClick and self.currentSprite: self.addBrick(x, y)
            if rightClick: self.removeBrick(x, y)

    def getCoordByViewPoint(self, x, y):
        view_x, view_y = self.viewPoint
        pos_x, pos_y = self.position
        x = (x - view_x) - (pos_x - view_x)
        y = (y - view_y) - (pos_y - view_y)
        return (x, y)

    def on_mouse_press(self, x, y, buttons, modifiers):
        x, y = self.getCoordByViewPoint(x, y)
        leftClick = buttons == 1
        rightClick = buttons == 4

        if self.palitra.visible:
            if leftClick: self.selectBrick(x, y)
        else:
            if leftClick and self.currentSprite: self.addBrick(x, y)
            if rightClick: self.removeBrick(x, y)

        sleep(0.01)

        self.updateInfo(str(x) + ',' + str(y))

    def updateInfo(self, info):
        info = "press: " + str(info)
        info += ", width: " + str(self.currentWidth) + ", height: " + str(self.currentHeight)
        if self.currentSprite: info += ", currentSprite: " + str(self.currentSprite.src)
        if self.currentType: info += ", currentType: " + str(self.currentType)

        self.text.element.text = info


    def selectBrick(self, x, y):
        dx, dy = self.palitra.position
        x = x - dx
        y = y - dy

        fakeObj = self.objectProvider.getFakeObject((x,y))
        selectedSprite = self.objectProvider.checkIntersecWithRightPanel(fakeObj)
        if selectedSprite:
            self.currentSprite = selectedSprite
        else:
            self.currentSprite = None

    def addBrick(self, x, y):
        x = x // 32 * 32 + 16
        y = y // 32 * 32 + 16

        spriteObj = sprite.Sprite(self.currentSprite.src)
        spriteObj.src = self.currentSprite.src
        spriteObj.position = (x, y)
        spriteObj.type = self.currentType
        spriteObj.cshape = cm.AARectShape(spriteObj.position,spriteObj.width//2,spriteObj.height//2)

        intersec = self.objectProvider.checkIntersec(spriteObj)
        if intersec: return

        self.walls.append(spriteObj)
        self.collision.add(spriteObj)
        self.add(spriteObj)

    def removeBrick(self, x, y):
        x = x // 32 * 32 + 16
        y = y // 32 * 32 + 16

        fakeObj = self.objectProvider.getFakeObject((x,y))
        collisions = self.collision.objs_colliding(fakeObj)
        if collisions:
            for wall in self.walls:
                if wall in collisions:
                    if wall in self.walls: self.walls.remove(wall)
                    if wall in self: self.remove(wall)
                    if wall in self.collision.objs: self.collision.remove_tricky(wall)

    #
    # def generateBackground(self):
    #     src = 'assets/202.jpg'
    #
    #     for x in range(16, 32*50, 32):
    #         for y in range(16, 32*50, 32):
    #             spriteObj = sprite.Sprite(src)
    #             spriteObj.src = src
    #             spriteObj.position = (x, y)
    #             spriteObj.type = 1
    #             spriteObj.cshape = cm.AARectShape(spriteObj.position,spriteObj.width//2,spriteObj.height//2)
    #             self.walls.append(spriteObj)
    #             self.collision.add(spriteObj)
    #             self.add(spriteObj)