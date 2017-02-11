import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

x = np.linspace(0, 2*np.pi)

fig, (ax1, ax2) = plt.subplots(2)
line, = ax1.plot(x, np.sin(x))
line2, = ax2.plot(x, np.cos(x))
iteration = ax1.text(0.1, 0.9, 'i=x',horizontalalignment='center',
                             verticalalignment='center',
                             transform=ax1.transAxes)
w = 1
def animate(t):
    line.set_data(x, np.sin(x-w*t))
    line2.set_data(x, np.cos(x-w*t))
    iteration.set_text(t)
    return [line, line2, iteration]

anim = animation.FuncAnimation(fig, animate, np.linspace(0,1))
anim.save("animation_test.mp4")
plt.show()
