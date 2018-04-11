class Ordinary(object):
    """Экземпляры этого класса могут дополняться атрибутами
    во время исполнения.
    # http://pythonz.net/references/named/object.__slots__/
    """


class WithSlots(object):

    __slots__ = 'static_attr'


a = Ordinary()
b = WithSlots()

a.__dict__ # {}
b.__dict__  # AttributeError

a.__weakref__  # None
b.__weakref__  # AttributeError

a.static_attr = 1
b.static_attr = 1

a.dynamic_attr = 2
b.dynamic_attr = 2  # AttributeError