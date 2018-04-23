from collections import OrderedDict

example_str = "Hello"
print(id(example_str))
example_str += ' Name!'
print(id(example_str))

example_str[start:stop:step]
example_str[::step]

print(example_str[::-1])
example_str.count('l')
example_str.capitalize()
example_str.isdigit()

template = "%s smth (%s) - %d"
template % ('s1', 's2', 534)

"{num} kb. ({author})".format(num=123, author='Bill')

subject = 'test'
author = 'Sam'

f"smth {subject}, - {author}"

num = 8
f"Binary: {num} is {num:#b}"
num = 2/3
print(f"{num:.3f}")

encoded_string = example_str.encode(encoding='utf-8')
decoded_string = encoded_string.decode()

from mymodule import multiply
dir(multiply)
print(multiply.__code__.co_code)
import dis
print(dis.dis(multiply))

ordered = OrderedDict()

#set
print({1,2,3} | {4,5,6})
print({1,2,3} & {3,4,5})
print({1,2,3} - {3,4,5})
print({1,2,3} ^ {3,4,5})
frozenset


def get_seconds():
    pass

get_seconds.__doc__
get_seconds.__name__

def add(x: int, y: int) -> int:
    pass


#list comprehencions

square_map = {number: number**2 for number in range(5) if number%2==0}



action = 'a'

if action == "a":
    add_dvd(db)
elif action == "e":
    edit_dvd(db)
elif action == "l":
    list_dvds(db)
elif action == "r":
    remove_dvd(db)
elif action == "i":
    import_(db)
elif action == "x":
    export(db)
elif action == "q":
    quit(db)

functions = dict(a=add_dvd, e=edit_dvd, l=list_dvds, r=remove_dvd, i=import_, x=export, q=quit)
functions[action](db)

call = {(".aix", "dom"): self.import_xml_dom,
        (".aix", "etree"): self.import_xml_etree,
        (".aix", "sax"): self.import_xml_sax,
        (".ait", "manual"): self.import_text_manual,
        (".ait", "regex"): self.import_text_regex,
        (".aib", None): self.import_binary,
        (".aip", None): self.import_pickle}
result = call[extension, reader](filename)




getattr(obj, name, val)
hasattr(obj, name)
setattr(obj, name, val)
vars(obj) #Возвращает контекст объекта obj в виде словаря или локальный контекст, если аргумент obj не определен

def get_function(module, function_name):
    function = get_function.cache.get((module, function_name), None)
    if function is None:
        try:
            function = getattr(module, function_name)
            if not hasattr(function, "__call__"):
                raise AttributeError()
            get_function.cache[module, function_name] = function
        except AttributeError:
            function = None
    return function
get_function.cache = {}