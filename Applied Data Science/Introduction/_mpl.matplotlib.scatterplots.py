import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9])
y = x

colors = ['green'] * (len(x)-1)
colors.append('red')

plt.figure()
plt.scatter(x,y, s=100, c=colors)
plt.show()

zip_generator = zip([1,2,3,4,5], [6,7,8,9,0])
print(list(zip_generator))

zip_generator = zip([1,2,3,4,5], [6,7,8,9,0])
x, y = zip(*zip_generator)
print(x)
print(y)

plt.xlabel('some Title')
plt.ylabel('some Title')
plt.title('some Title')

plt.legend(loc=4, frameon=False, title='Legend Title')