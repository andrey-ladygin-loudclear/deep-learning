class planer:
    count = 1
    _peoples = 0

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '<Class ' + str(self.__name__) + '>'

    def __del__(self): # should not be called (it is better)
        print('destructor')

    @property
    def workable_peoples(self):
        # you can calculate workable peoples in this method
        return self._peoples / 1.2


    peoples = property()

    @peoples.setter
    def peoples(self, value):
        self._peoples = max(value, 0)

    @peoples.getter
    def peoples(self):
        return self._peoples

    @peoples.deleter
    def peoples(self):
        print('remove all peoples on the planet')
        del self._peoples

    @classmethod
    def from_string(cls, name):
        return cls(name)

pl = planer('Earch')

print(pl.__dict__)

print(dict.fromkeys('123456'))
print(planer.__mro__)



class GetAttr(object):
    def __getattribute__(self, name):
        f = lambda: "Hello {}".format(name)
        return f

g = GetAttr()
g.Mark()
'Hello Mark'


# Третий способ (который, с точки зрения языка Python, является наи
# более правильным) заключается в следующем: if not isinstance(other,
# Point): return NotImplemented. В этом третьем случае, когда метод воз
# вращает NotImplemented, интерпретатор попытается вызвать метод
# other.__eq__(self), чтобы определить, поддерживает ли тип other срав
# нение с типом Point, и если в этом типе не будет обнаружен такой ме
# тод или он также возвращает NotImplemented, интерпретатор возбудит
# исключение TypeError.


class Ord:
    def __getattr__(self, char):
        return ord(char)

# ord.a вернет число 97,
# ord.Z вернет 90,
# ord.е вернет 229

class Const:
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise ValueError("cannot change a const attribute")
        self.__dict__[name] = value

    def __delattr__(self, name):
        if name in self.__dict__:
            raise ValueError("cannot delete a const attribute")
        raise AttributeError("'{0}' object has no attribute '{1}'"
                         .format(self.__class__.__name__, name))



# __delattr__(self, name) del x.n Удаляет атрибут n из объекта x
# __dir__(self) dir(x) Возвращает список имен атрибутов объекта x
# __getattr__(self, name) v = x.n Возвращает значение атрибута n объекта x, если он существует
# __getattribute__(self, name) v = x.n Возвращает значение атрибута n объекта x; подробности в тексте
# __setattr__(self, name, value) x.n = v Присваивает значение v атрибуту n объекта x


class Image:
    def __init__(self, width, height, filename="", background="#FFFFFF"):
        self.filename = filename
        self.__background = background
        self.__data = {}
        self.__width = width
        self.__height = height
        self.__colors = {self.__background}

    def __getattr__(self, name):
        if name == "colors":
            return set(self.__colors)

        classname = self.__class__.__name__

        if name in frozenset({"background", "width", "height"}):
            return self.__dict__["_{classname}__{name}".format(
                **locals())]

        raise AttributeError("'{classname}' object has no "
                                 "attribute '{name}'".format(**locals()))



# functor
class Strip:
    def __init__(self, characters):
        self.characters = characters

    def __call__(self, string):
        return string.strip(self.characters)

class SortKey:
    def __init__(self, *attribute_names):
        self.attribute_names = attribute_names

    def __call__(self, instance):
        values = []
        for attribute_name in self.attribute_names:
            values.append(getattr(instance, attribute_name))
        return values