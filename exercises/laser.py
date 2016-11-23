import numpy as np
import matplotlib.pyplot as plt
c = 1
NX = 400
X, dx = np.linspace(0, 1, NX, retstep=True)
dt = dx/c
NT = 200
I = range(NT)

Jyplus = np.zeros_like(X)
Jyminus = np.zeros_like(X)
Fplus = np.zeros_like(X)
Fminus = np.zeros_like(X)
# Fplus[0] = Fminus[0] = 1
def iterate(Fplus, Fminus, Jyplus, Jyminus):
    Fplus[1:] = Fplus[:-1] -0.25*dt * (Jyplus[:-1] + Jyminus[1:])
    Fminus[1:] = Fminus[:-1] -0.25*dt * (Jyplus[:-1] - Jyminus[1:])

Ey_history = np.empty((NT, NX))
Bz_history = np.empty((NT, NX))
LASER_PERIOD = 20*dt
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

fig, ax = plt.subplots()
Bline, Eline = ax.plot(X,Ey,X,Bz)

def animate(i):
    Bline.set_ydata(Bz_history[i])
    Eline.set_ydata(Ey_history[i])
    return [Bline, Eline]

from matplotlib import animation
anim = animation.FuncAnimation(fig, animate, I)
plt.show()
