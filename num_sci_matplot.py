import numpy as np

x = [2,3,4,5]
y = np.array(x)

print type(x), x
print type(y), y

print x[1:3]
print y[1:3]

print y[[0,2]]
print y[y>3] # y>3 return array of booleans

print y * 5 # increase each elements
print y ** 2 # increase each elements

matrix = [[1,2,4], [3, 1, 0]]
np_array = np.array(matrix)

print np.random.rand()
print np.random.randn()

from scipy import optimize

def f(x):
    return (x[0] - 3.2) ** 2 + (x[1] - 0.1) ** 2 + 3

print f([3.2, 0.1])
x_min = optimize.minimize(f, [5,5])
print x_min.x

from scipy import linalg

a = np.array([[3,2,0], [1,-1,0], [0,5,1]])
b = np.array([2, 4, -1])

x = linalg.solve(a, b)
print x
print np.dot(a, x)

X = np.random.rand(4, 3)
U, D, V = linalg.svd(X)
print U.shape, D.shape, V.shape
print type(U), type(D), type(V)

from matplotlib import pylab as plt

plt.plot([1,2,3,4], [1,4,9,16])
plt.show()

x = np.arange(-10, 10, 0.1)
y = x**3
plt.plot(x, y)
plt.show()

import numpy as np
from scipy import interpolate
x = np.arange(0, 10, 2)
y = np.exp(-x / 3.0)

f = interpolate.interp1d(x, y, kind='linear')
xnew = np.arange(0, 8, 0.1)
ynew = f(xnew)

plt.plot(x, y, 'o', xnew, ynew, '-')
plt.show()