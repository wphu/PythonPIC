import numpy as np
from Grid import Grid
from Species import Species
from pic3 import run
import plotting

def cold_plasma_oscillations(filename, plasma_frequency=1, qmratio=-1, dt=0.2, NT=150,
                             NG=32, N_electrons=128, L=2 * np.pi, epsilon_0=1,
                             push_amplitude=0.001, push_mode=1):

    """Implements cold plasma oscillations from Birdsall and Langdon

    (plasma excited by a single cosinusoidal mode via position displacements)"""
    print("Running cold plasma oscillations")
    particle_charge = plasma_frequency**2 * L / float(N_electrons * epsilon_0 * qmratio)
    particle_mass = particle_charge / qmratio

    g = Grid(L=L, NG=NG, NT=NT)
    electrons = Species(particle_charge, particle_mass, N_electrons, "electrons", NT=NT)
    list_species = [electrons]
    for species in list_species:
        species.distribute_uniformly(g.L)
        species.sinusoidal_position_perturbation(push_amplitude, push_mode, g.L)
    params = NT, dt, epsilon_0
    return run(g, list_species, params, filename)

if __name__ == '__main__':
    cold_plasma_oscillations("data_analysis/CO1.hdf5")
    plotting.plotting("data_analysis/CO1.hdf5")
