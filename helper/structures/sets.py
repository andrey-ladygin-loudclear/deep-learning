# The main use-cases where sets are a good choice are membership tests
# (testing if an element is present in the collection) and, unsurprisingly,
# set operations such as union, difference, and intersection.

# create a list that contains duplicates
x = list(range(1000)) + list(range(500))
# the set *x_unique* will contain only
# the unique elements in x
x_unique = set(x)

# Code              Time
# s.union(t)        O(S + T)
# s.intersection(t) O(min(S, T))
# s.difference(t)   O(S)


docs = ["the cat is under the table",
        "the dog is under the table",
        "cats and dogs smell roses",
        "Carla eats an apple"]

# Building an index using sets
index = {}
for i, doc in enumerate(docs):
    # We iterate over each term in the document
    for word in doc.split():
        # We build a set containing the indices
        # where the term appears
        if word not in index:
            index[word] = {i}
        else:
            index[word].add(i)

# Querying the documents containing both "cat" and "table"
index['cat'].intersection(index['table'])



{ x for x in range(10)} # Генератор множеств
set([1, 3, 3, 2]) == {1, 2, 3}
set((i*2 for i in range(10))) == {i*2 for i in range(10)}