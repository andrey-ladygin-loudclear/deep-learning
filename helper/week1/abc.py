# ABC           Наследует       API         Примеры
# Callable      object          ()          Все функции, методы и лямбда-функции
# Container     object          in          bytearray, bytes, dict, frozenset, list, set, str, tuple
# Hashable      object          hash()      bytes, frozenset, str, tuple
# Iterable      object          iter()      bytearray, bytes, collections.deque, dict, frozenset, list, set, str, tuple

# Iterator      Iterable        iter(), next()
# Sized         object          len()       bytearray, bytes, collections.deque, dict, frozenset, list, set, str, tuple

# Mapping       Container,      ==, !=, [], len(), iter(), in,          dict
#               Iterable,       get(), items(), keys(), values()
#               Sized


# MutableMapping Mapping        ==, !=, [], del, len(), iter(), in, clear(), get(),                         dict
#                               items(), keys(), pop(), popitem(), setdefault(), update(), values()

# Sequence      Container,      [], len(), iter(), reversed(), in, count(), index()                         bytearray, bytes, list, str, tuple
#               Iterable,
#               Sized


# MutableSequence Container,    [], +=, del, len(), iter(), reversed(), in, append(),                       bytearray, list
#                 Iterable,     count(), extend(), index(), insert(), pop(), remove(), reverse()
#                 Sized

# Set           Container,      <, <=, ==, !=, =>, >, &, |, ^,          frozenset, set
#               Iterable,       len(), iter(), in, isdisjoint()
#               Sized

# MutableSet    Set             <, <=, ==, !=, =>, >, &, |, ^,          set
#                               &=, |=, ^=, =, len(), iter(),
#                               in, add(), clear(), discard(),
#                               isdisjoint(), pop(), remove()
