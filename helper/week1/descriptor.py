class CachedXmlShadow:
    def __init__(self, attribute_name):
        self.attribute_name = attribute_name
        self.cache = {}

    def __get__(self, instance, owner=None):
        xml_text = self.cache.get(id(instance))
        if xml_text is not None:
            return xml_text
        return self.cache.setdefault(id(instance), xml.sax.saxutils.escape(getattr(instance, self.attribute_name)))

# Ключи необходимы, потому
# что дескрипторы создаются для всего класса, а не для его экземпля
# ров. (Метод dict.setdefault() возвращает значение для заданного клю
# ча или, если элемента с таким ключом нет, создает новый элемент с за
# данным ключом и значением и возвращает значение, что весьма удоб
# но для нас.)

class XmlShadow:
    def __init__(self, attribute_name):
        self.attribute_name = attribute_name

    def __get__(self, instance, owner=None):
        return xml.sax.saxutils.escape(getattr(instance, self.attribute_name))

class Product:
    __slots__ = ("__name", "__description", "__price")

    name_as_xml = XmlShadow("name")
    description_as_xml = XmlShadow("description")

    def __init__(self, name, description, price):
        self.__name = name
        self.description = description
        self.price = price

product = Product("Chisel <3cm>", "Chisel & cap", 45.25)
print(product.name, product.name_as_xml, product.description_as_xml)
#('Chisel <3cm>', 'Chisel &lt;3cm&gt;', 'Chisel &amp; cap')


########################################################################################################
class ExternalStorage:
    __slots__ = ("attribute_name",)
    __storage = {}

    def __init__(self, attribute_name):
        self.attribute_name = attribute_name

    def __set__(self, instance, value):
        self.__storage[id(instance), self.attribute_name] = value

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return self.__storage[id(instance), self.attribute_name]

class Point:
    __slots__ = ()
    # Определив пустой кортеж в качестве значения атрибута __slots__, мы
    # тем самым гарантируем, что класс вообще не будет иметь никаких ат
    # рибутов данных.
    # При попытке выполнить присваивание атрибуту
    # self.x интерпретатор обнаружит наличие дескриптора с именем «x»
    # и вызовет его метод __set__().

    x = ExternalStorage("x")
    y = ExternalStorage("y")

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

########################################################################################################
########################################################################################################
########################################################################################################
class Property:
    def __init__(self, getter, setter=None):
        self.__getter = getter
        self.__setter = setter
        self.__name__ = getter.__name__

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return self.__getter(instance)

    def __set__(self, instance, value):
        if self.__setter is None:
            raise AttributeError("'{0}' is read-only".format(self.__name__))
        return self.__setter(instance, value)

class NameAndExtension:
    def __init__(self, name, extension):
        self.__name = name
        self.extension = extension

    @Property # Задействуется нестандартный дескриптор Property
    def name(self):
        return self.__name

    @Property # Задействуется нестандартный дескриптор Property
    def extension(self):
        return self.__extension

    @extension.setter # Задействуется нестандартный дескриптор Property
    def extension(self, extension):
        self.__extension = extension

