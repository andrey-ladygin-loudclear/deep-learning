from cmath import sin, exp
import numpy as np
from matplotlib import pylab as plt
from scipy import linalg
from scipy import optimize


def f(x):
    return (sin(x / 5.0) * exp(x / 10.0) + 5 * exp(-x / 2.0)).real

def h(x):
    return int(f(x))

x = [i for i in np.arange(1, 30, .001)]
y = [h(i) for i in np.arange(1, 30, .001)]

#plt.plot(x, y, 'ro')

# plt.plot(x, y)
# plt.show()

# for c in range(100):
#     print c, optimize.minimize(h, c, method='BFGS').fun
#
# plt.plot(x, y)
# plt.show()

# print(optimize.minimize(h, 30, method='BFGS'))
print(optimize.differential_evolution(h, [(1,30)]))

# print(optimize.differential_evolution(h, [(1,30)]))
# result = optimize.minimize(h, 2)
# print(result)