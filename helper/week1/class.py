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