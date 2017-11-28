class planer:
    count = 1

    def __init__(self, name):
        self.name = name


    def __repr__(self):
        return '<Class ' + str(self.__name__) + '>'

    def __del__(self): # should not be called (it is better)
        print('destructor')

pl = planer('Earch')

print(pl.__dict__)
