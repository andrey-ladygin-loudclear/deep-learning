import cocos
import pyglet
from pyglet.window import key
from scene import Scene

global keyboard, scroller

cocos.director.director.init(autoscale=False, resizable=True, width=2000, height=800)

keyboardHandler = key.KeyStateHandler()
scrollerHandler = cocos.layer.ScrollingManager()
cocos.director.director.window.push_handlers(keyboardHandler)
#cocos.director.director.window.push_handlers(scroller)

sceneHandler = Scene(keyboardHandler, scrollerHandler)
scrollerHandler.add(sceneHandler)

scene = cocos.scene.Scene(sceneHandler, scrollerHandler)
scene.schedule(sceneHandler.checkButtons)

cocos.director.director.on_resize = sceneHandler.resize

cocos.director.director.run(scene)
