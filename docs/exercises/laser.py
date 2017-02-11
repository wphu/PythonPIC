"""first attempts at electromagnetic field solver (local!)"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from Simulation import Simulation
from Grid import Grid
from Species import Species

g = Grid(L = 1, NG=100, epsilon_0=1, c= 1)
NT = 400
I = range(NT)

Ey_history = np.empty((NT, g.NG))
Bz_history = np.empty((NT, g.NG))


LASER_PERIOD = 50*g.dt
for i in I:
    t = i * g.dt
    # drive boundary condition
    laser_phase = 2*np.pi*t/(LASER_PERIOD)
    B0 = np.cos(laser_phase)
    E0 = np.sin(laser_phase)
    # turn fields into $F_{\pm}$ for solving
    g.apply_laser_BC(B0, E0)
    # Unroll $F_{\pm}$ into physical fields B, E
    Ey_history[i], Bz_history[i] = g.unroll_EyBz()
    g.iterate_EM_field()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
field_line, = ax.plot(g.x,Ey_history[0],Bz_history[0])
Bline, = ax.plot(g.x, np.zeros_like(g.x), Bz_history[0])
Eline, = ax.plot(g.x, Ey_history[0], np.zeros_like(g.x))
axis_line, = ax.plot(g.x, np.zeros_like(g.x), np.zeros_like(g.x), "k--")
ax.grid()
ax.set_ylim(Ey_history.min(), Ey_history.max())
ax.set_zlim(Bz_history.min(), Bz_history.max())
ax.set_xlabel("x")
ax.set_ylabel("Ey")
ax.set_zlabel("Bz")

def animate(i):
    Bline.set_xdata(g.x)
    Eline.set_xdata(g.x)
    field_line.set_xdata(g.x)

    Eline.set_ydata(Ey_history[i])
    field_line.set_ydata(Ey_history[i])

    Eline.set_3d_properties(np.zeros_like(g.x))
    Bline.set_3d_properties(Bz_history[i])
    field_line.set_3d_properties(Bz_history[i])
    return [Bline, field_line]

anim = animation.FuncAnimation(fig, animate, I, interval = 0.1)
plt.show()
