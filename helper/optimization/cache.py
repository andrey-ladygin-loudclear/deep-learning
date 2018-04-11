from functools import lru_cache


@lru_cache()
def sum2(a, b):
    print("Calculating {} + {}".format(a, b))
    return a + b


print(sum2(1, 2))
# Output:
# Calculating 1 + 2
# 3

print(sum2(1, 2))
# Output:
# 3



import timeit
setup_code = '''
from functools import lru_cache
from __main__ import fibonacci
fibonacci_memoized = lru_cache(maxsize=None)(fibonacci)
'''
results = timeit.repeat('fibonacci_memoized(20)',
                        setup=setup_code,
                        repeat=1000,
                        number=1)
print("Fibonacci took {:.2f} us".format(min(results)))
# Output: Fibonacci took 0.01 us


# you can use joblib
