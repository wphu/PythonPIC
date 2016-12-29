import numpy as np
from Grid import Grid
from Species import Species
from pic3 import run_electromagnetic
import plotting

def hybrid_oscillations(filename, plasma_frequency=1, qmratio=-1, dt=0.2, NT=300,
                             NG=32, N_electrons=128, L=2 * np.pi, epsilon_0=1,
                             push_amplitude=0.001, push_mode=1, v0=0.0,
                             cyclotron_frequency=1):
    """Implements hybrid oscillations from Birdsall and Langdon"""
    print("Running hybrid oscillations")
    particle_charge = plasma_frequency**2 * L / float(2*N_electrons * epsilon_0 * qmratio)
    particle_mass = particle_charge / qmratio

    g = Grid(L=L, NG=NG, NT=NT)
    electrons1 = Species(particle_charge, particle_mass, N_electrons, "electrons", NT=NT)
    electrons1.v[:,0] = v0
    list_species = [electrons1]
    for i, species in enumerate(list_species):
        species.distribute_uniformly(g.L, 0.5*g.dx*i)
        species.sinusoidal_position_perturbation(push_amplitude, push_mode, g.L)
        species.sinusoidal_velocity_perturbation(1,
            -cyclotron_frequency*push_amplitude,
            push_mode,
            g.L)
    params = NT, dt, epsilon_0, cyclotron_frequency/qmratio
    return run_electromagnetic(g, list_species, params, filename)

if __name__ == '__main__':
    hybrid_oscillations("data_analysis/HO1.hdf5")
    plotting.plotting("data_analysis/HO1.hdf5", lines=True)
