import numpy as np

N = int(128)
NT = 150
dt = 0.2
plasma_frequency = 1
qmratio = -1
L = 2 * np.pi
NG = 32
epsilon_0 = 1
push_amplitude = 0.001
push_mode = 1

# calculations
T = NT * dt
particle_charge = plasma_frequency**2 * L / (N * epsilon_0 * qmratio)
particle_mass = particle_charge / qmratio
density = N / L
plasma_frequency = np.sqrt(density * particle_charge**2 / particle_mass / epsilon_0)
dt = dt / plasma_frequency
