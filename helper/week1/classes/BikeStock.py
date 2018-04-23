import struct


class Bike:

    def __init__(self, identity, name, quantity, price):
        assert len(identity) > 3, ("invalid bike identity '{0}'"
                                   .format(identity))
        self.__identity = identity
        self.name = name
        self.quantity = quantity
        self.price = price


    @property
    def identity(self):
        "The bike's identity"
        return self.__identity


    @property
    def name(self):
        "The bike's name"
        return self.__name

    @name.setter
    def name(self, name):
        assert len(name), "bike name must not be empty"
        self.__name = name


    @property
    def quantity(self):
        "How many of this bike are in stock"
        return self.__quantity

    @quantity.setter
    def quantity(self, quantity):
        assert 0 <= quantity, "quantity must not be negative"
        self.__quantity = quantity


    @property
    def price(self):
        "The bike's price"
        return self.__price

    @price.setter
    def price(self, price):
        assert 0.0 <= price, "price must not be negative"
        self.__price = price


    @property
    def value(self):
        "The value of these bikes in stock"
        return self.quantity * self.price

_BIKE_STRUCT = struct.Struct("<8s30sid")

class BikeStock:
    def __init__(self, filename):
        self.__file = BinaryRecordFile.BinaryRecordFile(filename,
                                                        _BIKE_STRUCT.size)
        self.__index_from_identity = {}
        for index in range(len(self.__file)):
            record = self.__file[index]
        if record is not None:
            bike = _bike_from_record(record)
        self.__index_from_identity[bike.identity] = index


def _bike_from_record(record):
    ID, NAME, QUANTITY, PRICE = range(4)
    parts = list(_BIKE_STRUCT.unpack(record))
    parts[ID] = parts[ID].decode("utf8").rstrip("\x00")
    parts[NAME] = parts[NAME].decode("utf8").rstrip("\x00")
    return Bike(*parts)


def _record_from_bike(bike):
    return _BIKE_STRUCT.pack(bike.identity.encode("utf8"),
                             bike.name.encode("utf8"),
                             bike.quantity, bike.price)


class BikeStock:

    def __init__(self, filename):
        self.__file = BinaryRecordFile.BinaryRecordFile(filename,
                                                        _BIKE_STRUCT.size)
        self.__index_from_identity = {}
        for index in range(len(self.__file)):
            record = self.__file[index]
            if record is not None:
                bike = _bike_from_record(record)
                self.__index_from_identity[bike.identity] = index


    def close(self):
        "Compacts and closes the file"
        self.__file.inplace_compact()
        self.__file.close()


    def append(self, bike):
        "Adds a new bike to the stock"
        index = len(self.__file)
        self.__file[index] = _record_from_bike(bike)
        self.__index_from_identity[bike.identity] = index


    def __delitem__(self, identity):
        "Deletes the stock record for the specified bike"
        del self.__file[self.__index_from_identity[identity]]
        del self.__index_from_identity[identity]


    def __getitem__(self, identity):
        "Retrieves the stock record for the specified bike"
        record = self.__file[self.__index_from_identity[identity]]
        return None if record is None else _bike_from_record(record)


    def __change_bike(self, identity, what, value):
        index = self.__index_from_identity[identity]
        record = self.__file[index]
        if record is None:
            return False
        bike = _bike_from_record(record)
        if what == "price" and value is not None and value >= 0.0:
            bike.price = value
        elif what == "name" and value is not None:
            bike.name = value
        else:
            return False
        self.__file[index] = _record_from_bike(bike)
        return True

    change_name = lambda self, identity, name: self.__change_bike(
        identity, "name", name)
    change_name.__doc__ = "Changes the bike's name"

    change_price = lambda self, identity, price: self.__change_bike(
        identity, "price", name)
    change_price.__doc__ = "Changes the bike's price"


    def __change_stock(self, identity, amount):
        index = self.__index_from_identity[identity]
        record = self.__file[index]
        if record is None:
            return False
        bike = _bike_from_record(record)
        bike.quantity += amount
        self.__file[index] = _record_from_bike(bike)
        return True

    increase_stock = (lambda self, identity, amount:
                      self.__change_stock(identity, amount))
    increase_stock.__doc__ = ("Increases the stock held for the "
                              "specified bike by by the given amount")

    decrease_stock = (lambda self, identity, amount:
                      self.__change_stock(identity, -amount))
    decrease_stock.__doc__ = ("Decreases the stock held for the "
                              "specified bike by by the given amount")


    def __iter__(self):
        for index in range(len(self.__file)):
            record = self.__file[index]
            if record is not None:
                yield _bike_from_record(record)


if __name__ == "__main__":
    bicycles = BikeStock.BikeStock(bike_file)
    value = 0.0
    for bike in bicycles:
        value += bike.value
    bicycles.increase_stock("GEKKO", 2)
    for bike in bicycles:
        if bike.identity.startswith("B4U"):
            if not bicycles.increase_stock(bike.identity, 1):
            print("stock movement failed for", bike.identity)