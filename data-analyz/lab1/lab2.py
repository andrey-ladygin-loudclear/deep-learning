from cmath import sin, exp
import numpy as np
from scipy import linalg


def f(x):
    return sin(x / 5.0) * exp(x / 10.0) + 5 * exp(-x / 2.0)

matrix = np.array([
    [1, 1],
    [1, 15]
])
res = np.array([f(1), f(15)])
print linalg.solve(matrix,res)

matrix = np.array([
    [1, 1, 1],
    [1, 8, 8**2],
    [1, 15, 15**2]
])
res = np.array([f(1), f(8), f(15)])
print linalg.solve(matrix,res)

matrix = np.array([
    [1, 1, 1, 1],
    [1, 4, 4**2, 4**3],
    [1, 10, 10**2, 10**3],
    [1, 15, 15**2, 15**3],
])
res = np.array([f(1), f(4), f(10), f(15)])
answer = linalg.solve(matrix,res)

map(lambda x: x.round(2), answer)