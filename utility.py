import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

n = 1
for arg in sys.argv:
    if (n == 2):
        nbPoint = int(arg)
    n += 1

print(nbPoint)
points = []

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Affichage des points')


def onpick(event):
    if len(points) < nbPoint:
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        point = tuple(zip(xdata[ind], ydata[ind]))
        points.append(point)
        print('onpick point:', point)
    else:
        print('already have {} points'.format(len(points)))


#while (nbPoint > len(points)):
fig.canvas.mpl_connect('pick_event', onpick)
plt.show()


np.interp()