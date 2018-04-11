# Heaps are data structures designed to quickly find and extract the maximum (or minimum)
# value in a collection. A typical use-case for heaps is to process a series of incoming tasks in
# order of maximum priority.


# One can theoretically use a sorted list using the tools in the bisect module; however, while
# extracting the maximum value will take O(1) time (using list.pop), insertion will still take
# O(N) time (remember that, even if finding the insertion point takes O(log(N)) time, inserting
# an element in the middle of a list is still a O(N) operation). A heap is a more efficient data
# structure that allows for insertion and extraction of maximum values with O(log(N)) time
# complexity.

import heapq
collection = [10, 3, 3, 4, 5, 6]
heapq.heapify(collection)

heapq.heappop(collection)
# Returns: 3
heapq.heappush(collection, 1)



from queue import PriorityQueue
queue = PriorityQueue()
for element in collection:
    queue.put(element)
queue.get()
# Returns: 3



queue = PriorityQueue()
queue.put((3, "priority 3"))
queue.put((2, "priority 2"))
queue.put((1, "priority 1"))
queue.get()
# Returns: (1, "priority 1")
