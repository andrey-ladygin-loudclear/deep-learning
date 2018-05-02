import cocos
import pyglet
from cocos.layer import MultiplexLayer
from pyglet.window import key
from scene import Scene

from game.map.menu import MainMenu, MapOptionMenu, ScoreMenu

global keyboard, scroller
#https://github.com/los-cocos/cocos/blob/master/samples/demo_grid_effects.py
cocos.director.director.init(autoscale=False, resizable=True, width=2000, height=800)

keyboardHandler = key.KeyStateHandler()
scrollerHandler = cocos.layer.ScrollingManager()
cocos.director.director.window.push_handlers(keyboardHandler)

sceneHandler = Scene(keyboardHandler, scrollerHandler)
scrollerHandler.add(sceneHandler)

menulayer = MultiplexLayer(MainMenu(), MapOptionMenu(sceneHandler.map), ScoreMenu())
sceneHandler.setMenuLayer(menulayer)

scene = cocos.scene.Scene(sceneHandler, scrollerHandler, menulayer)
scene.transform_anchor = (0, 0)
sceneHandler.transform_anchor = (0, 0)
scene.schedule(sceneHandler.checkButtons)

cocos.director.director.on_resize = sceneHandler.resize

cocos.director.director.run(scene)
