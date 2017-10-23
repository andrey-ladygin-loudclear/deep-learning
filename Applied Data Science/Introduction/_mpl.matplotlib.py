import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

print(mpl.get_backend())

plt.plot(3, 2, '-')
plt.plot(3, 5, '.')
plt.show()

fig = Figure()
canvas = FigureCanvasAgg(fig)

ax = fig.add_subplot(111)
ax.plot(3,2, '.')
canvas.print_png('test.png')

plt.figure()
plt.plot(3,2,'o')
ax = plt.gca()
ax.axis([-5,6,0,10])
plt.show()