# Dictionaries are implemented as hash maps and are very good at element insertion,
# deletion, and access; all these operations have an average O(1) time complexity.

# In Python versions up to 3.5, dictionaries are unordered collections.

# Access, insertion, and removal of an item in a dictionary scales as O(1) with the size of the
# dictionary.
# ! However, note that the computation of the hash function still needs to happen
# and, for strings, the computation scales with the length of the string.

def counter_dict(items):
    counter = {}
    for item in items:
        if item not in counter:
            counter[item] = 0
    else:
        counter[item] += 1
    return counter

# can be simplified to

from collections import defaultdict
def counter_defaultdict(items):
    counter = defaultdict(int)
    for item in items:
        counter[item] += 1
    return counter

# or
items = []
from collections import Counter
counter = Counter(items)

# Code                       N=1000 (μs) N=2000 (μs) N=3000 (μs) Time
# Counter(items)             51.48       96.63       140.26      O(N)
# counter_dict(items)        111.96      197.13      282.79      O(N)
# counter_defaultdict(items) 120.90      238.27      359.60      O(N)


docs = ["the cat is under the table",
        "the dog is under the table",
        "cats and dogs smell roses",
        "Carla eats an apple"]

matches = [doc for doc in docs if "table" in doc] # search all sentences with table word


# inverted index O(1)

index = {}
for i, doc in enumerate(docs):
    for word in doc.split():
        if word not in index:
            index[word] = [i]
        else:
            index[word].append(i)

print(index)
# {'the': [0, 0, 1, 1], 'cat': [0], 'is': [0, 1], 'under': [0, 1], 'table': [0, 1],
# 'dog': [1], 'cats': [2], 'and': [2], 'dogs': [2], 'smell': [2], 'roses': [2], 'Carla': [3],
# 'eats': [3], 'an': [3], 'apple': [3]}

results = index['table']
result_documents = [docs[i] for i in results]


import re
words = re.findall(r'\w+', open('hamlet.txt').read().lower())


{a:a**2 for a in range(1, 10)} # dict generator
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81}


sum[value] = sum.get(value, 0) + 1