import numpy as np
import constants
N = int(128)
NG = 32
NT = 150
L = 2*np.pi
dt = 0.2
particle_charge = -1
particle_mass = 1
epsilon_0 = constants.epsilon_0

density = N/L
plasma_frequency = np.sqrt(density*particle_charge**2/particle_mass/epsilon_0)
print("Plasma frequency: {} rad/s".format(plasma_frequency))
# dt = 0.01/(plasma_frequency/np.pi/2)
dt = 0.2/plasma_frequency
# print("dt: {}".format(dt))

T = NT*dt

#TODO: could use an init.py
x, dx = np.linspace(0,L,NG, retstep=True,endpoint=False)
charge_density = np.zeros_like(x)

position_shift = L/N/10
x_particles = np.linspace(0,L,N, endpoint=False) + L/N/10
print("position shift: {}".format(position_shift))

# wave initialization: for two stream instability
# x_particles += 0.001*np.sin(x_particles*2*np.pi/L)
v_particles = np.zeros(N)
# v_particles[::2] = -1
