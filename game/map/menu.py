
import pyglet
from cocos.director import director
from cocos.layer import MultiplexLayer
from cocos.menu import Menu, MenuItem, zoom_in, zoom_out, ToggleMenuItem, shake, shake_back, CENTER, BOTTOM, RIGHT, LEFT


class MainMenu(Menu):

    def __init__(self, main_scene):
        super(MainMenu, self).__init__("GROSSINI'S SISTERS")

        pyglet.font.add_directory('./assets')
        self.font_title['font_name'] = 'You Are Loved'
        self.font_title['font_size'] = 72
        self.font_item['font_name'] = 'You Are Loved'
        self.font_item_selected['font_name'] = 'You Are Loved'
        self.menu_valign = CENTER
        self.menu_halign = CENTER
        self.main_scene = main_scene

        items = []

        items.append(MenuItem('Generate new background', self.generate_new_background))
        items.append(MenuItem('Options', self.on_options))
        #items.append(MenuItem('Scores', self.on_scores))
        items.append(MenuItem('Quit', self.on_quit))

        self.create_menu(items, zoom_in(), zoom_out())

    def generate_new_background(self):
        self.main_scene.map.generate_new_background()

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
        self.menu_halign = LEFT

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

