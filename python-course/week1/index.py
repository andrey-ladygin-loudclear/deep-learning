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