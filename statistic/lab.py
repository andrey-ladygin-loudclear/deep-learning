import random

import numpy as np
import scipy.stats as st
from matplotlib import pylab as plt

#x = np.random.choice([1,2,3,4,5,6], 100)
y = []

Xm = 5.
k = 1.
N = 800

def f_parto(x):
    return (k / x**(k+1))
    #return (k * Xm**k / x**(k+1.))

for x in range(N):
    try:
        if x >= Xm:
            y.append(f_parto(x))
    except ZeroDivisionError:
        pass

pareto = st.pareto(k, loc=0, scale=.1)
stats = pareto.rvs(size=N)
# plt.plot(stats)
#plt.plot(y)
#plt.hist(np.random.randn(1000), 50)

plt.plot(y)
plt.hist(stats, 200, normed=True, facecolor='green', alpha=0.5)

xmin,xmax,ymin,ymax = (-10, 1400, -0.0001, 0.005)
plt.axis([xmin,xmax,ymin,ymax])

#print stats
plt.show()