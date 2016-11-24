"""first attempts at electromagnetic field solver (local!)"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

c = 1
NX = 100
X, dx = np.linspace(0, 1, NX, retstep=True)
dt = dx/c
NT = 400
I = range(NT)

Jyplus = np.zeros_like(X)
Jyminus = np.zeros_like(X)
Fplus = np.zeros_like(X)
Fminus = np.zeros_like(X)

def iterate(Fplus, Fminus, Jyplus, Jyminus):
    """
    calculate Fplus, Fminus in next iteration based on their previous
    values

    assumes fixed left ([0]) boundary condition

    F_plus(n+1, j) = F_plus(n, j) - 0.25 * dt * (Jyminus(n, j-1) + Jplus(n, j))
    F_minus(n+1, j) = F_minus(n, j) - 0.25 * dt * (Jyminus(n, j+1) - Jplus(n, j))

    TODO: check viability of laser BC
    take average of last term instead at last point instead

    """
    Fplus[1:] = Fplus[:-1] -0.25*dt * (Jyplus[:-1] + Jyminus[1:])
    Fminus[1:-1] = Fminus[0:-2] -0.25*dt * (Jyplus[2:] - Jyminus[1:-1])

    # boundary condition
    Fminus[-1] = Fminus[-2] -0.25*dt * (Jyplus[0] - Jyminus[-1])

Ey_history = np.empty((NT, NX))
Bz_history = np.empty((NT, NX))

# Jyplus = np.exp(-(X-0.5)**2/16)
# Jyminus = -Jyplus

LASER_PERIOD = 50*dt
for i in I:
    t = i * dt
    # drive boundary condition
    laser_phase = 2*np.pi*t/(LASER_PERIOD)
    B0 = np.cos(laser_phase)
    E0 = np.sin(laser_phase)
    # turn fields into $F_{\pm}$ for solving
    Fplus[0] = (E0 + B0)/2
    Fminus[0] = (E0 - B0)/2

    # Unroll $F_{\pm}$ into physical fields B, E
    Ey = Fplus + Fminus
    Bz = Fplus - Fminus
    Ey_history[i] = Ey
    Bz_history[i] = Bz

    iterate(Fplus, Fminus, Jyplus, Jyminus)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
field_line, = ax.plot(X,Ey,Bz)
Bline, = ax.plot(X, np.zeros_like(X), Bz)
Eline, = ax.plot(X, Ey, np.zeros_like(X))
axis_line, = ax.plot(X, np.zeros_like(X), np.zeros_like(X), "k--")
ax.grid()

def animate(i):
    Bline.set_xdata(X)
    Eline.set_xdata(X)
    field_line.set_xdata(X)

    Eline.set_ydata(Ey_history[i])
    field_line.set_ydata(Ey_history[i])

    Eline.set_3d_properties(np.zeros_like(X))
    Bline.set_3d_properties(Bz_history[i])
    field_line.set_3d_properties(Bz_history[i])
    return [Bline, field_line]

anim = animation.FuncAnimation(fig, animate, I, interval = 0.1)
plt.show()
