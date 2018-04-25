class History:
    __slots__ = ("attribute_name",)
    __storage = []

    def __init__(self, attribute_name):
        self.attribute_name = attribute_name

    def __set__(self, instance, value):
        self.__storage.append(value)

    def __get__(self, instance, owner=None):
        return self.__storage

class Text():
    #__slots__ = ("text",)
    text = ''
    history = History("text")

    def __init__(self, text = ''):
        self.text = text

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        #self.__dict__[key] = value
        super().__setattr__('history', value)

        #setattr(self, 'history', value) # cause recursion
        #self.history = value # cause recursion

        #print(self.history.attribute_name) # ERROR
        #print(getattr(self.history.__slots__)) # ERROR



poema = Text('some poema')
poema2 = Text('some poema')
print(poema.text)
poema.text = 'new some poema'
poema2.text = 'new some poema 2'
print(poema.text)
print(poema.history)
print(poema2.history)
print(Text.history)
