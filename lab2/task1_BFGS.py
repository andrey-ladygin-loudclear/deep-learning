from cmath import sin, exp
import numpy as np
from matplotlib import pylab as plt
from scipy import linalg
from scipy import optimize


def f(x):
    return (sin(x / 5.0) * exp(x / 10.0) + 5 * exp(-x / 2.0)).real

def c(arr):
    return sum([f(i) for i in arr])

#x = [i for i in np.arange(-10, 10, 1)]
#y = [f(i) for i in np.arange(-10, 10, 1)]

# plt.plot(x, y)
# plt.show()

print(optimize.minimize(f, 2, method='BFGS'))
print(optimize.minimize(f, 30, method='BFGS'))
# result = optimize.minimize(f, 2)
# print(result)