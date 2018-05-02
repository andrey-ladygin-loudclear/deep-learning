import os
from time import sleep

import cocos
from cocos import sprite
from cocos.layer import Layer, director, ScrollableLayer, pyglet
from cocos.text import Label
import cocos.collision_model as c
from copy import copy
from pyglet.window import key

from palitra import Palitra
from map import Map


class CurrentBlockDescriptor:
    def __init__(self):
        self.block = None

    def __get__(self, instance, owner=None):
        return self.block

    def __set__(self, instance, value=None):
        if self.block:
            instance.remove(self.block)

        if value:
            self.block = copy(value)
            self.update_position(instance.position)
            instance.add(self.block, z=2)
        else:
            self.block = None

    def update_position(self, position):
        x, y = position
        self.block.position = (-x, -y + 650)


class Scene(ScrollableLayer):
    anchor = (0, 0)
    is_event_handler = True
    walls = []
    labels = []
    #collision = cm.CollisionManagerBruteForce()
    #palitraCollision = cm.CollisionManagerBruteForce()

    focusX = 1500
    focusY = 500

    currentBlock = CurrentBlockDescriptor()

    appendMode = 1

    buttonsTextHelp = "q-pallet, 1 - background, 2 - unmovable background, 3 - indestructible object, 4 - object, t - increase type"

    viewPoint = (0, 0)
    windowWidth = 2000
    windowHeight = 800

    def __init__(self, keyboard, scroller):
        super().__init__()
        self.set_view(0, 0, self.windowWidth, self.windowHeight, 0, 0)
        self.palitra = Palitra()
        self.map = Map()
        self.add(self.map, z=2)
        self.add(self.palitra, z=3)

        self.keyboard = keyboard
        self.scroller = scroller
        # self.buttonsProvider = ButtonsProvider()
        # self.objectProvider = ObjectProvider(self.keyboard, self.collision, self.palitraCollision)
        #
        # self.helperLayer = cocos.layer.Layer()
        # self.buttonsInfo = Label(self.buttonsTextHelp, font_name='Helvetica', font_size=12, anchor_x='left',  anchor_y='top')
        # self.text = Label("Some text", font_name='Helvetica', font_size=12, anchor_x='left',  anchor_y='bottom')
        # self.helperLayer.add(self.text)
        # self.helperLayer.add(self.buttonsInfo)
        # self.add(self.helperLayer, z=5)
        #
        # self.palitra = cocos.layer.Layer()
        # self.palitraObject = []
        # self.add(self.palitra, z=2)

    def setMenuLayer(self, menu_layer):
        self.menu_layer = menu_layer
        self.menu_layer.visible = False

    def checkButtons(self, dt):
        x_direction = self.keyboard[key.LEFT] - self.keyboard[key.RIGHT]
        y_direction = self.keyboard[key.DOWN] - self.keyboard[key.UP]
        print(self.keyboard[key.DOWN], self.keyboard[key.UP])
        x, y = self.position

        if self.keyboard[key.Q]:
            self.palitra.visible = 1 - self.palitra.visible
            sleep(0.2)

        if self.keyboard[key.E]:
            self.menu_layer.visible = 1 - self.menu_layer.visible
            sleep(0.2)

        if x_direction:
            x += x_direction * 20

        if y_direction:
            y += y_direction * 20

        if x_direction or y_direction:
            self.set_view(0, 0, self.windowWidth, self.windowHeight, x, y)
            self.palitra.updatePosition(self.windowWidth, self.windowHeight, self.position)

        if self.keyboard[key.SPACE]:
            self.set_view(0, 0, self.windowWidth, self.windowHeight, 0, 0)
            self.palitra.updatePosition(self.windowWidth, self.windowHeight, self.position)


    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        x, y = self.getCoordByViewPoint(x, y)
        leftClick = buttons == 1
        rightClick = buttons == 4

        if not self.palitra.visible:
            if leftClick and self.currentBlock: self.map.addBrick(x, y, self.currentBlock)
            if rightClick: self.map.removeBrick(x, y)

    def on_mouse_press(self, x, y, buttons, modifiers):
        if self.palitra.visible:
            self.currentBlock = self.palitra.click(x, y)

        x, y = self.getCoordByViewPoint(x, y)
        leftClick = buttons == 1
        rightClick = buttons == 4

        if not self.palitra.visible:
            if leftClick and self.currentBlock: self.map.addBrick(x, y, self.currentBlock)
            if rightClick: self.map.removeBrick(x, y)

        sleep(0.01)

        #self.updateInfo(str(x) + ',' + str(y))

    def resize(self, width, height):
        self.viewPoint = (width // 2, height // 2)
        self.windowWidth = width
        self.windowHeight = height
        self.palitra.resizeMap(width, height, self.viewPoint)

        #self.buttonsInfo.position = (0, self.currentHeight)
        #self.setLayersPosition()
        #self.resizeMap()

    def getCoordByViewPoint(self, x, y):
        view_x, view_y = self.viewPoint
        pos_x, pos_y = self.position
        x = (x - view_x) - (pos_x - view_x)
        y = (y - view_y) - (pos_y - view_y)
        return (x, y)