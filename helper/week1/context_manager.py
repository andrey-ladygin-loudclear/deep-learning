import contextlib

try:
    with contextlib.nested(open(source), open(target, "w")) as (fin, fout):
        for line in fin:
            pass
except EOFError:
    pass


try:
    with AtomicList(items) as atomic:
        atomic.append(58289)
        del atomic[3]
        atomic[8] = 81738
        atomic[index] = 38172
except (AttributeError, IndexError, ValueError) as err:
    print("no changes applied:", err)

# Если в ходе выполнения операций никаких исключений не возникло,
# все операции будут применены к оригинальному списку (items), но ес
# ли возникло исключение, список останется без изменений. Ниже при
# водится реализация менеджера контекста AtomicList:

class AtomicList:
    def __init__(self, alist, shallow_copy=True):
        self.original = alist
        self.shallow_copy = shallow_copy

    def __enter__(self):
        self.modified = (self.original[:] if self.shallow_copy else copy.deepcopy(self.original))
        return self.modified

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:# check on exception
            self.original[:] = self.modified

        # if return True it cause raise Exception if exc_type is None