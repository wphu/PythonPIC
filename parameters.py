import numpy as np
N = int(1e4)
NG = 32
NT = 100
L = 1
dt = 0.001
T = NT*dt
particle_charge = 1
particle_mass = 1

#TODO: could use an init.py
x, dx = np.linspace(0,L,NG, retstep=True,endpoint=False)
charge_density = np.zeros_like(x)
x_particles = np.linspace(0,L,N, endpoint=False) + L/N/100
x_particles += 0.001*np.sin(x_particles*2*np.pi/L)
v_particles = np.zeros(N)
# v_particles[::2] = -1
