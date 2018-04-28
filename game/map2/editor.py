import cocos
import pyglet
from pyglet.window import key
from mouseInput import MouseInput

global keyboard, scroller

cocos.director.director.init(autoscale=False, resizable=True, width=2000, height=800)

keyboardHandler = key.KeyStateHandler()
scrollerHandler = cocos.layer.ScrollingManager()
cocos.director.director.window.push_handlers(keyboardHandler)
#cocos.director.director.window.push_handlers(scroller)

mouseInputHandler = MouseInput(keyboardHandler, scrollerHandler)
scrollerHandler.add(mouseInputHandler)

scene = cocos.scene.Scene(mouseInputHandler, scrollerHandler)
scene.schedule(mouseInputHandler.checkButtons)

cocos.director.director.on_resize = mouseInputHandler.resize

cocos.director.director.run(scene)
