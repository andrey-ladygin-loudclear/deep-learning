import numpy as np

A = np.array([[1,2,3,4,5],
              [2,3,4,5,6],
              [3,4,5,6,7]])

total = A.sum(axis=0)
print(total)
percentage = 100*A/total.reshape(1,5)
print(percentage)
