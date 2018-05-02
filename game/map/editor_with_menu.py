import cocos
import pyglet
from cocos.director import director
from cocos.layer import MultiplexLayer
from cocos.menu import Menu, MenuItem, zoom_in, zoom_out, ToggleMenuItem, shake, shake_back, CENTER, BOTTOM, RIGHT, LEFT
from pyglet.window import key
from scene import Scene




class MainMenu(Menu):

    def __init__(self):

        # call superclass with the title
        super(MainMenu, self).__init__("GROSSINI'S SISTERS")

        pyglet.font.add_directory('.')

        # you can override the font that will be used for the title and the items
        self.font_title['font_name'] = 'You Are Loved'
        self.font_title['font_size'] = 72

        self.font_item['font_name'] = 'You Are Loved'
        self.font_item_selected['font_name'] = 'You Are Loved'

        # you can also override the font size and the colors. see menu.py for
        # more info

        # example: menus can be vertical aligned and horizontal aligned
        self.menu_valign = CENTER
        self.menu_halign = CENTER

        items = []

        items.append(MenuItem('New Game', self.on_new_game))
        items.append(MenuItem('Options', self.on_options))
        items.append(MenuItem('Scores', self.on_scores))
        items.append(MenuItem('Quit', self.on_quit))

        self.create_menu(items, zoom_in(), zoom_out())

    # Callbacks
    def on_new_game(self):
        #        director.set_scene(StartGame())
        print("on_new_game()")

    def on_scores(self):
        self.parent.switch_to(2)

    def on_options(self):
        self.parent.switch_to(1)

    def on_quit(self):
        director.pop()


class OptionMenu(Menu):

    def __init__(self):
        super(OptionMenu, self).__init__("GROSSINI'S SISTERS")

        self.font_title['font_name'] = 'You Are Loved'
        self.font_title['font_size'] = 72

        self.font_item['font_name'] = 'You Are Loved'
        self.font_item_selected['font_name'] = 'You Are Loved'

        self.menu_valign = BOTTOM
        self.menu_halign = RIGHT

        items = []
        items.append(MenuItem('Fullscreen', self.on_fullscreen))
        items.append(ToggleMenuItem('Show FPS: ', self.on_show_fps, True))
        items.append(MenuItem('OK', self.on_quit))
        self.create_menu(items, shake(), shake_back())

    # Callbacks
    def on_fullscreen(self):
        director.window.set_fullscreen(not director.window.fullscreen)

    def on_quit(self):
        self.parent.switch_to(0)

    def on_show_fps(self, value):
        director.show_FPS = value


class ScoreMenu(Menu):

    def __init__(self):
        super(ScoreMenu, self).__init__("GROSSINI'S SISTERS")

        self.font_title['font_name'] = 'You Are Loved'
        self.font_title['font_size'] = 72
        self.font_item['font_name'] = 'You Are Loved'
        self.font_item_selected['font_name'] = 'You Are Loved'

        self.menu_valign = BOTTOM
        self.menu_halign = LEFT

        self.create_menu([MenuItem('Go Back', self.on_quit)])

    def on_quit(self):
        self.parent.switch_to(0)







global keyboard, scroller
#https://github.com/los-cocos/cocos/blob/master/samples/demo_grid_effects.py
cocos.director.director.init(autoscale=False, resizable=True, width=2000, height=800)

keyboardHandler = key.KeyStateHandler()
scrollerHandler = cocos.layer.ScrollingManager()
cocos.director.director.window.push_handlers(keyboardHandler)
#cocos.director.director.window.push_handlers(scroller)

mapScene = Scene(keyboardHandler, scrollerHandler)
menulayer = MultiplexLayer(MainMenu(), OptionMenu(), ScoreMenu())

scrollerHandler.add(mapScene)

scene = cocos.scene.Scene(mapScene, scrollerHandler, menulayer)
scene.schedule(mapScene.checkButtons)

cocos.director.director.on_resize = mapScene.resize

cocos.director.director.run(scene)
