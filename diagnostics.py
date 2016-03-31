import numpy as np
epsilon_0 = 1
def energies(r, v, m, dx, rho, phi):
    particle_kinetic_energy = 0.5 * m * np.sum(v * v)
    field_potential_energy = 0.5 * epsilon_0 * dx * np.sum(rho*phi)
    total_energy = particle_kinetic_energy + field_potential_energy
    return particle_kinetic_energy, field_potential_energy, total_energy
