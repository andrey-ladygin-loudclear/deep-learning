class Image:
    def __init__(self, width, height, filename="", background="#FFFFFF"):
        self.filename = filename
        self.__background = background
        self.__data = {}
        self.__width = width
        self.__height = height
        self.__colors = {self.__background}

    @property
    def width(self):
        return self.__width

    # def __getattr__(self, name):
    #     if name == "colors":
    #         return set(self.__colors)
    #
    #     classname = self.__class__.__name__
    #
    #     print('__getattr__', classname, name)
    #
    #     if name in frozenset({"background", "width", "height"}):
    #         return self.__dict__["_{classname}__{name}".format(
    #             **locals())]
    #
    #     raise AttributeError("'{classname}' object has no "
    #                          "attribute '{name}'".format(**locals()))

class Png(Image):
    def __init__(self, width, height, filename="", background="#FFFFFF"):
        super().__init__(width, height, filename, background)
        self.type = 'png'
        self.__type = '__png'

im = Image(20, 10, 'test.txt')
print(im.__dict__)
png = Png(20, 10, 'test.txt')
print(png.__dict__)
print(png.width)