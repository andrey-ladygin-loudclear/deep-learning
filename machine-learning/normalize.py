import numpy as np

X = np.array([
    [2104, 1416, 1534, 852],
    [5, 3, 3, 2],
    [1, 2, 2, 1],
    [45, 40, 30, 36],
])

Y = np.array([460, 232, 315, 178])
ONES = [[1],[1],[1],[1]]
#X[row, column]
#X[:,0] - first column

X = np.append(ONES, X, 1)

X_transpose = X.transpose()

X_dot = np.dot(X_transpose, X)

X_inv = np.linalg.pinv(X_dot)

X_inv_transpose = np.dot(X_inv, X_transpose)

res = np.dot(X_inv_transpose, Y)

print res