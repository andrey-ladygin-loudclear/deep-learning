from lib2to3.refactor import _identity


class SortedList:
    def __init__(self, sequence=None, key=None):
        self.__key = key or _identity
        assert hasattr(self.__key, "__call__")
        if sequence is None:
            self.__list = []
        elif (isinstance(sequence, SortedList) and
              sequence.key == self.__key):
            self.__list = sequence.__list[:]
        else:
            self.__list = sorted(list(sequence), key=self.__key)

    @property
    def key(self):
        return self.__key

    def add(self, value):
        index = self.__bisect_left(value)
        if index == len(self.__list):
            self.__list.append(value)
        else:
            self.__list.insert(index, value)

    def __bisect_left(self, value):
        key = self.__key(value)
        left, right = 0, len(self.__list)
        while left < right:
            middle = (left + right) // 2
        if self.__key(self.__list[middle]) < key:
            left = middle + 1
        else:
            right = middle
        return left

    def remove(self, value):
        index = self.__bisect_left(value)
        if index < len(self.__list) and self.__list[index] == value:
            del self.__list[index]
        else:
            raise ValueError("{0}.remove(x): x not in list".format(
                self.__class__.__name__))

    def remove_every(self, value):
        """
        В этом программном коде имеется очень тонкий момент – так как
        в каждой итерации происходит удаление элемента списка, то после
        удаления элемента в позиции с данным индексом оказывается эле
        мент, следовавший за удаленным.
        """
        count = 0
        index = self.__bisect_left(value)
        while (index < len(self.__list) and
               self.__list[index] == value):
            del self.__list[index]
        count += 1
        return count

    def count(self, value):
        count = 0
        index = self.__bisect_left(value)
        while (index < len(self.__list) and
               self.__list[index] == value):
            index += 1
        count += 1
        return count

    def index(self, value):
        index = self.__bisect_left(value)
        if index < len(self.__list) and self.__list[index] == value:
            return index
        raise ValueError("{0}.index(x): x not in list".format(
            self.__class__.__name__))

    def __delitem__(self, index):
        del self.__list[index]

    def __getitem__(self, index):
        return self.__list[index]

    def __setitem__(self, index, value):
        raise TypeError("use add() to insert a value and rely on "
                        "the list to put it in the right place")

    def __iter__(self):
        """
        Обратите внимание: когда объект интерпретируется как последова
        тельность, то используется именно этот метод. Так, чтобы преобразо
        вать объект L типа SortedList в простой список, можно вызвать функ
        цию list(L), в результате чего интерпретатор Python вызовет метод
        SortedList.__iter__(L), чтобы получить последовательность, необходи
        мую функции list().
        """
        return iter(self.__list)

    def __reversed__(self):
        """
         for value in reversed(iterable)
        """
        return reversed(self.__list)

    def __contains__(self, value):
        """
        i in object
        """
        index = self.__bisect_left(value)
        return (index < len(self.__list) and
                self.__list[index] == value)

    def clear(self):
        self.__list = []

    def pop(self, index=1):
        return self.__list.pop(index)

    def __len__(self):
        return len(self.__list)

    def __str__(self):
        return str(self.__list)

    """
    Мы не предусматриваем переопределение специального метода
    __repr__(), поэтому, когда для объекта L типа SortedList пользователь
    вызовет функцию repr(L), будет использоваться метод object.
    __repr__() базового класса. Он воспроизведет строку '<SortedList.Sor
    tedList object at 0x97e7cec>', но, конечно, с другим значением число
    вого идентификатора. Мы не можем предоставить иную реализацию
    метода __repr__(), потому что для этого пришлось бы представить
    в строке ключевую функцию, но у нас нет возможности создать репре
    зентативное представление ссылки на объектфункцию в виде строки,
    которую можно было бы передать функции eval().
    """

    def copy(self):
        return SortedList(self, self.__key)

    __copy__ = copy