def loop():
    res = []
    for i in range(100000):
        res.append(i * i)
    return sum(res)


def comprehension():
    return sum([i * i for i in range(100000)])


def generator():
    return sum(i * i for i in range(100000))


# %timeit loop()
# 100 loops, best of 3: 16.1 ms per loop
# %timeit comprehension()
# 100 loops, best of 3: 10.1 ms per loop
# %timeit generator()
# 100 loops, best of 3: 12.4 ms per loop
