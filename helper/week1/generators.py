# Функция-генератор, или метод-генератор – это функция, или
# метод, содержащая выражение yield. В результате обращения
# к функции генератору возвращается итератор. Значения из ите-
# ратора извлекаются по одному, с помощью его метода __next__().
# При каждом вызове метода __next__() он возвращает результат
# вычисления выражения yield. (Если выражение отсутствует,
# возвращается значение None.) Когда функция-генератор завер-
# шается или выполняет инструкцию return, возбуждается исклю-
# чение StopIteration.
# На практике очень редко приходится вызывать метод __next__()
# или обрабатывать исключение StopIteration. Обычно функция-
# генератор используется в качестве итерируемого объекта. Ни-
# же приводятся две практически эквивалентные функции. Функ-
# ция слева возвращает список, а функция справа возвращает ге-
# нератор.
# # Создает и возвращает список
import sys


def letter_range(a, z):
    result = []
    while ord(a) < ord(z):
        result.append(a)
        a = chr(ord(a) + 1)
    return result

# Возвращает каждое
# значение по требованию
def letter_range(a, z):
    while ord(a) < ord(z):
        yield a
        a = chr(ord(a) + 1)
list(letter_range("m", "v"))

# Результаты, воспроизводимые обеими функциями, можно обойти
# с помощью цикла for, например for letter in letter_range("m",
# "v"):. Однако когда требуется получить список символов с помо-
# щью функции слева, достаточно просто вызвать ее как let
# ter_range("m", "v"), а для функции справа необходимо выпол-
# нить преобразование: list(letter_range("m", "v")).
# Функции-генераторы и методы-генераторы (а также выраже-
# ния-генераторы) более полно рассматриваются в главе 8.


def items_in_key_order(d):
    for key in sorted(d):
        yield key, d[key]
# equivalent =>
def items_in_key_order(d):
    return ((key, d[key]) for key in sorted(d))


# Выражение yield поочередно возвращает каждое значение вызываю
# щей программе. Кроме того, если будет вызван метод send() генерато
# ра, то переданное значение будет принято функциейгенератором в ка
# честве результата выражения yield. Ниже показано, как можно ис
# пользовать новую функциюгенератор:

def quarters(next_quarter=0.0):
    while True:
        received = (yield next_quarter)
        if received is None:
            next_quarter += 0.25
        else:
            next_quarter = received

result = []
generator = quarters()
while len(result) < 5:
    x = next(generator)
    if abs(x - 0.5) < sys.float_info.epsilon:
        x = generator.send(1.0)
    result.append(x)




def positive_result(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        assert result >= 0, function.__name__ + "() result isn't >= 0"
        return result
    #wrapper.__name__ = function.__name__
    #wrapper.__doc__ = function.__doc__
    return wrapper

def bounded(minimum, maximum):
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            if result < minimum:
                return minimum
            elif result > maximum:
                return maximum
            return result
        return wrapper
    return decorator



if __debug__:
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(os.path.join(
        tempfile.gettempdir(), "logged.log"))
    logger.addHandler(handler)
    def logged(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            log = "called: " + function.__name__ + "("
            log += ", ".join(["{0!r}".format(a) for a in args] +
                     ["{0!s}={1!r}".format(k, v) for k, v in kwargs.items()])
            result = exception = None
            try:
                result = function(*args, **kwargs)
                return result
            except Exception as err:
                exception = err
            finally:
                log += ((") > " + str(result)) if exception is None
                else ") {0}: {1}".format(type(exception),
                                     exception))
                logger.debug(log)
            if exception is not None:
                raise exception
        return wrapper
else:
    def logged(function):
        return function



def is_unicode_punctuation(s : str) -> bool:
    for c in s:
        if unicodedata.category(c)[0] != "P":
            return False
    return True



def strictly_typed(function):
    annotations = function.__annotations__
    arg_spec = inspect.getfullargspec(function)
    assert "return" in annotations, "missing type for return value"
    for arg in arg_spec.args + arg_spec.kwonlyargs:
        assert arg in annotations, ("missing type for parameter '" +
                                    arg + "'")
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        for name, arg in (list(zip(arg_spec.args, args)) +
                          list(kwargs.items())):
            assert isinstance(arg, annotations[name]), (
                "expected argument '{0}' of {1} got {2}".format(
                    name, annotations[name], type(arg)))
        result = function(*args, **kwargs)
        assert isinstance(result, annotations["return"]), (
            "expected return of {0} got {1}".format(
                annotations["return"], type(result)))
        return result
    return wrapper
