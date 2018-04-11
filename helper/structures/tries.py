from random import choice
from string import ascii_uppercase


def random_string(length):
    """Produce a random string made of *length* uppercase ascii
    characters"""
    return ''.join(choice(ascii_uppercase) for i in range(length))


strings = [random_string(32) for i in range(10000)]
matches = [s for s in strings if s.startswith('AA')]
print(matches)



from patricia import trie
strings_dict = {s:0 for s in strings}
# A dictionary where all values are 0
strings_trie = trie(**strings_dict)
matches = list(strings_trie.iter('AA'))
print(matches)
# If you look closely, the timing for this input size is 60.1 μs, which is about 30 times
# faster (1.76 ms = 1760 μs) than linear search!
# Note that if we want to return all the prefixes that match, the running time will be
# proportional to the number of results that match the prefix. Therefore, when designing
# timing benchmarks, care must be taken to ensure that we are always returning the same
# number of results.
# The scaling properties of a trie versus a linear scan for datasets of different sizes that
# contains ten prefix matches are shown in the following table:

# Algorithm     N=10000 (μs)    N=20000 (μs)    N=30000 (μs)    Time
# Trie          17.12           17.27           17.47           O(S)
# Linear scan   1978.44         4075.72         6398.06         O(N)

#
# other C-optimized trie libraries are also available, such as datrie and marisa-trie.
