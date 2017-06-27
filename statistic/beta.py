import scipy.stats as st
from matplotlib import pylab as plt

x_coordinates = []
y_coordinates = []

A = 5
B = 1
N = 10000


def BetaFn(x):
    return st.beta.pdf(x, A, B)

for x in range(N):
    x = float(x) / N
    y = BetaFn(x)
    # print "X = %s, Y = %s" % (x, y)
    x_coordinates.append(x)
    y_coordinates.append(y)

fig = plt.figure()
fig.suptitle('Beta Function', fontsize=20)
plt.xlabel('N', fontsize=18)
plt.ylabel('Value', fontsize=16)

stats = st.beta.rvs(A, B, size=N)
plt.hist(stats, 100, normed=True, facecolor='green', alpha=0.5)
plt.plot(x_coordinates, y_coordinates)
plt.show()