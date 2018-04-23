import os

_DELETED = b"\x01"
_OKAY = b"\x02"

#https://github.com/BaneZhang/python/blob/master/Programming_in_Python3/Examples/BinaryRecordFile.py
class BinaryRecordFile:

    def __init__(self, filename, record_size, auto_flush=True):
        """A random access binary file that behaves rather like a list
        with each item a bytes or bytesarray object of record_size.
        """
        self.__record_size = record_size + 1 # with state byte
        mode = "w+b" if not os.path.exists(filename) else "r+b"
        self.__fh = open(filename, mode)
        self.auto_flush = auto_flush


    @property
    def record_size(self):
        "The size of each item"
        return self.__record_size - 1


    @property
    def name(self):
        "The name of the file"
        return self.__fh.name


    def flush(self):
        """Flush writes to disk
        Done automatically if auto_flush is True
        """
        self.__fh.flush() # write programing buffer to disk or to OS buffer
        #os.fsync # for write OS buffer to disk


    def close(self):
        self.__fh.close()


    def __setitem__(self, index, record):
        """Sets the item at position index to be the given record
        The index position can be beyond the current end of the file.
        """
        assert isinstance(record, (bytes, bytearray)), \
            "binary data required"
        assert len(record) == self.record_size, (
            "record must be exactly {0} bytes".format(
                self.record_size))
        self.__fh.seek(index * self.__record_size)
        self.__fh.write(_OKAY)
        self.__fh.write(record)
        if self.auto_flush:
            self.__fh.flush()


    def __getitem__(self, index):
        """Returns the item at the given index position
        If there is no item at the given position, raises an
        IndexError exception.
        If the item at the given position has been deleted returns
        None.
        """
        self.__seek_to_index(index)
        state = self.__fh.read(1)
        if state != _OKAY:
            return None
        return self.__fh.read(self.record_size)


    def __seek_to_index(self, index):
        if self.auto_flush:
            self.__fh.flush()
        self.__fh.seek(0, os.SEEK_END)
        end = self.__fh.tell()
        offset = index * self.__record_size
        if offset >= end:
            raise IndexError("no record at index position {0}".format(
                index))
        self.__fh.seek(offset)


    def __delitem__(self, index):
        """Deletes the item at the given index position.
        See undelete()
        """
        self.__seek_to_index(index)
        state = self.__fh.read(1)
        if state != _OKAY:
            return
        self.__fh.seek(index * self.__record_size)
        self.__fh.write(_DELETED)
        if self.auto_flush:
            self.__fh.flush()


    def undelete(self, index):
        """Undeletes the item at the given index position.
        If an item is deleted it can be undeleted---providing compact()
        (or inplace_compact()) has not been called.
        """
        self.__seek_to_index(index)
        state = self.__fh.read(1)
        if state == _DELETED:
            self.__fh.seek(index * self.__record_size)
            self.__fh.write(_OKAY)
            if self.auto_flush:
                self.__fh.flush()
            return True
        return False


    def __len__(self):
        """The number number of record positions.
        This is the maximum number of records there could be at
        present. The true number may be less because some records
        might be deleted. After calling compact() (or
        inplace_compact()), this returns the true number.
        """
        if self.auto_flush:
            self.__fh.flush()
        self.__fh.seek(0, os.SEEK_END)
        end = self.__fh.tell()
        return end // self.__record_size


    def compact(self, keep_backup=False):
        """Eliminates blank and deleted records"""
        compactfile = self.__fh.name + ".$$$"
        backupfile = self.__fh.name + ".bak"
        self.__fh.flush()
        self.__fh.seek(0)
        fh = open(compactfile, "wb")
        while True:
            data = self.__fh.read(self.__record_size)
            if not data:
                break
            if data[:1] == _OKAY:
                fh.write(data)
        fh.close()
        self.__fh.close()

        os.rename(self.__fh.name, backupfile)
        os.rename(compactfile, self.__fh.name)
        if not keep_backup:
            os.remove(backupfile)
        self.__fh = open(self.__fh.name, "r+b")
    # Инструкция if data[:1] == _OKAY: таит в себе одну хит
    # рость. Оба объекта – и объект data и объект _OKAY – явля
    # ются объектами типа bytes
    # Нам необходимо сравнить
    # первый байт (один байт) объекта data с объектом _OKAY.
    # Когда к объекту типа bytes применяется операция среза,
    # возвращается объект bytes, но когда извлекается единст
    # венный байт, например, data[0], возвращается объект
    # типа int – значение байта. Поэтому здесь сравниваются
    # 1байтовый срез объекта data (его первый байт, байт со
    # стояния) с 1байтовым объектом _OKAY



    def inplace_compact(self):
        """Eliminates blank and deleted records in-place preserving the
        original order
        """
        index = 0
        length = len(self)
        while index < length:
            self.__seek_to_index(index)
            state = self.__fh.read(1)
            if state != _OKAY:
                for next in range(index + 1, length):
                    self.__seek_to_index(next)
                    state = self.__fh.read(1)
                    if state == _OKAY:
                        self[index] = self[next]
                        del self[next]
                        break
                else:
                    break
            index += 1
        self.__seek_to_index(0)
        state = self.__fh.read(1)
        if state != _OKAY:
            self.__fh.truncate(0)
        else:
            limit = None
            for index in range(len(self) - 1, 0, -1):
                self.__seek_to_index(index)
                state = self.__fh.read(1)
                if state != _OKAY:
                    limit = index
                else:
                    break
            if limit is not None:
                self.__fh.truncate(limit * self.__record_size)
        self.__fh.flush()


# Таблица 7.4. Методы и атрибуты объекта файла
# Синтаксис Описание
# f.close() Закрывает объект файла f и записывает в атрибут
# f.closed значение True
# f.closed Возвращает True, если файл закрыт
# f.encoding Кодировка, используемая при преобразованиях bytes ↔ str
# f.fileno() Возвращает дескриптор файла. (Доступно только для объектов файлов, имеющих дескрипторы.)
# f.flush() Выталкивает выходные буферы объекта f на диск
# f.isatty() Возвращает True, если объект файла ассоциирован с консолью. (Доступно только для объектов файлов,
    # ссылающихся на фактические файлы.)

# f.mode Режим, в котором был открыт объект файла f
# f.name Имя файла (если таковое имеется)
# f.newlines Виды последовательностей перевода строки, встречающиеся в текстовом файле f
# f.__next__() Возвращает следующую строку из объекта файла f. В большинстве случаев этот метод вызывается неявно,
    # например, for line in f f.peek(n) Возвращает n байтов без перемещения позиции указателя в файле

# f.read(count) Читает до count байтов из объекта файла f.
    # Если значение count не определено, то читаются все байты, начиная от текущей позиции и до конца.
    # При чтении в двоичном режиме возвращает объект bytes, при чтении
    # в текстовом режиме – объект str. Если из ничего не было прочитано (конец файла), возвращается пустой объект bytes или str

# f.readable() Возвращает True, если f был открыт для чтения
# f.readinto(ba) Читает до len(ba) байтов в объект ba типа bytearray
    # и возвращает число прочитанных байтов (0, если был
    # достигнут конец файла). (Доступен только в двоичном
    # режиме.)
# f.readline(count) Читает следующую строку (до count байтов, если значе
    # ние count было определено и число прочитанных байтов
    # было достигнуто раньше, чем встретился символ пере
    # вода строки \n), включая символ перевода строки \n

# f.readlines(sizehint) Читает все строки до конца файла и возвращает их в ви
    # де списка. Если значение аргумента sizehint определе
    # но, то будет прочитано примерно sizehint байтов, если
    # внутренние механизмы, на которые опирается объект
    # файла, поддерживают такую возможность

# f.seek(offset, whence)
    # Перемещает позицию указателя в файле (откуда будет
    # начато выполнение следующей операции чтения или за
    # писи) в заданное смещение, если аргумент whence не оп
    # ределен или имеет значение os.SEEK_SET. Перемещает по
    # зицию указателя в файле в заданное смещение (которое
    # может быть отрицательным) относительно текущей по
    # зиции, если аргумент whence имеет значение os.SEEK_CUR,
    # или относительно конца файла, если аргумент whence
    # имеет значение os.SEEK_END. Запись всегда выполняется
    # в конец файла, если был определен режим добавления
    # в конец "a", независимо от местоположения указателя
    # в файле. В текстовом режиме в качестве смещений
    # должны использоваться только значения, возвращае
    # мые методом tell()

# f.seekable() Возвращает True, если f поддерживает возможность произвольного доступа
# f.tell() Возвращает текущую позицию указателя в файле относительно его начала
# f.truncate(size) Усекает файл до текущей позиции указателя в файле или до размера size, если аргумент size задан
# f.writable() Возвращает True, если f был открыт для записи
# f.write(s) Записывает в файл объект s типа bytes/bytearray, если он был открыт в двоичном режиме, и объект s типа str, если он был открыт в текстовом режиме
# f.writelines(seq) Записывает в файл последовательность объектов (строки – для текстовых файлов, строки байтов – для двоичных файлов)
