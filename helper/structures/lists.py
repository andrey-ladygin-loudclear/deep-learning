# Lists shine in accessing, modifying, and appending elements. Accessing or modifying an
# element involves fetching the object reference from the appropriate position of the
# underlying array and has complexity O(1).

# Once all the slots are occupied,
# the list needs to increase the size of its underlying array,
# thus triggering a memory reallocation that can take O(N) time.

# Nevertheless, those memory allocations are infrequent

# The list operations that may have efficiency problems are those that add or remove elements
# at the beginning (or somewhere in the middle) of the list.

# When an item is inserted, or removed, from the beginning of a list,
# all the subsequent elements of the array
# need to be shifted by a position, thus taking O(N) time.

# Code              N=10000 (μs) N=20000 (μs) N=30000 (μs)  Time
# list.pop()        0.50         0.59         0.58          O(1)
# list.pop(0)       4.20         8.36         12.09         O(N)
# list.append(1)    0.43         0.45         0.46          O(1)
# list.insert(0, 1) 6.20         11.97        17.41         O(N)


# Deques, in addition to pop and append, expose the popleft and
# appendleft methods that have O(1) running time:

# Code                N=10000 (μs)  N=20000 (μs) N=30000 (μs) Time
# deque.pop()         0.41          0.47         0.51         O(1)
# deque.popleft()     0.39          0.51         0.47         O(1)
# deque.append(1)     0.42          0.48         0.50         O(1)
# deque.appendleft(1) 0.38          0.47         0.51         O(1)

# Searching for an item in a list is generally a O(N) operation and is performed using the
# list.index method. A simple way to speed up searches in lists is to keep the array sorted
# and perform a binary search using the bisect module.

import bisect
collection = [1, 2, 4, 5, 6]
bisect.bisect(collection, 3) # This function uses the binary search algorithm that has O(log(N)) running time.
# Result: 2

# Code                  N=10000 (μs)    N=20000 (μs)    N=30000 (μs)    Time
# list.index(a)         87.55           171.06          263.17          O(N)
# index_bisect(list, a) 3.16            3.20            4.71            O(log(N))


input = list(range(10))
for i, _ in enumerate(input): # generator
    input[i] += 1