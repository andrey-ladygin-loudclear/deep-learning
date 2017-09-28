import random


def gener(n=1000):
    for i in range(n):
        yield i


def batch(start, end, batch_i, batch_size):
    n_batches = (end - start) // batch_size
    print('start: ', start)
    print('end: ', end)
    print('batch_i: ', batch_i)
    print('batch_size: ', batch_size)
    print('n_batches: ', n_batches)

    if batch_i:
        batch_i -= 1

    if batch_i + 1 > n_batches:
        batch_i = batch_i % n_batches

    print('batch_i %: ', batch_i)

    start = start + batch_i * batch_size
    end = start + batch_size

    print('start %: ', start)
    print('end %: ', end)

    for i in gener():
        if i >= start and i < end:
            print(i)

        if i >= end: return


def rand_batch(batch_size, total, probability=0.5):
    count = 0
    print('batch_size: ', batch_size)
    for i in gener(total):
        if random.random() < probability:
            count += 1
            #print('i: ', i)
            yield i

            if count >= batch_size:
                return
    for i in rand_batch(batch_size - count, total, probability):
        yield i


for i in rand_batch(10, 3, 0.2):
    print(i)