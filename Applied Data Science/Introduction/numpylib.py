import numpy as np

mylist = [1,2,3]
x = np.array(mylist)


m = np.array([[7,8,9], [4,3,2]])

print(m.shape)

n = np.arange(0, 30, 2)
n.reshape(3, 5)

o = np.linspace(0, 4, 16)
# print(o)

o.resize(4,4)
# print(o)

np.ones((2,2))
np.zeros((2,2))

diagonal = np.eye(5)
#print(diagonal)

y = np.array([4,5,6])
ydiag = np.diag(y)
#print(ydiag)

p = np.ones([2,3])
np.vstack([p, 2*p])
np.hstack([p, 2*p])

x = np.array([y, y**2])
print(x.shape)

x.dtype
ff = x.astype('f')

y.max()
y.argmax()

y.min()
y.argmin()

#last 4 elements
y[-4:]

x[x > 30] = 30

test = np.random.randint(0, 10, (3,4))
test2 = test**2

for i, row in enumerate(test):
    print(i, row)

for i, j in zip(test, test2):
    print(i,'+',j,'=',i+j)