import numpy as np

Z = np.dot(w.T, X) + b

A = sigma(Z)

DZ = A - Y

db = 1/m * np.sum(DZ)
dw = 1/m * X * DZ.T

w = w - alpha * dw
b = b - alpha * db