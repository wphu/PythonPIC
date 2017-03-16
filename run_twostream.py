""" Run two stream instability"""
# coding=utf-8
import numpy as np

import plotting
from Constants import Constants
from Grid import Grid
from Simulation import Simulation
from Species import Species


def two_stream_instability(filename, plasma_frequency=1, qmratio=-1, dt=0.2, NT=300,
                             NG=32, N_electrons=128, L=2 * np.pi, epsilon_0=1,
                             push_amplitude=0.001, push_mode=1, v0=1.0, vrandom = 0):
    """Implements two stream instability from Birdsall and Langdon"""
    print("Running two stream instability")
    particle_charge = plasma_frequency**2 * L / float(2*N_electrons * epsilon_0 * qmratio)
    particle_mass = particle_charge / qmratio

    grid = Grid(L=L, NG=NG, NT=NT)
    k0 = 2*np.pi/L
    w0 = plasma_frequency
    print("k0*v0/w0 is", k0*v0/w0, "which means the regime is", "stable" if k0*v0/w0 > 2**0.5 else "unstable")
    electrons1 = Species(particle_charge, particle_mass, N_electrons, "beam1", NT=NT)
    electrons2 = Species(particle_charge, particle_mass, N_electrons, "beam2", NT=NT)
    electrons1.v[:] = v0
    electrons2.v[:] = -v0
    list_species = [electrons1, electrons2]
    for i, species in enumerate(list_species):
        species.distribute_uniformly(L, 0.5*grid.dx*i)
        species.sinusoidal_position_perturbation(push_amplitude, push_mode, grid.L)
        if vrandom > 0:
            species.random_velocity_perturbation(0, vrandom)
    params = NT, dt, epsilon_0
    run = Simulation(NT, dt, Constants(1, epsilon_0),
                     grid, list_species, filename=filename, title="Twostream instability")
    run.grid_species_initialization()
    run.run()
    return run

if __name__ == '__main__':
    np.random.seed(0)
    simulations = [
        two_stream_instability("data_analysis/TS1/TS1.hdf5",
                               NG=64,
                               N_electrons=512,
                               ),
        two_stream_instability("data_analysis/TS2/TS2.hdf5",
                               NG=64,
                               N_electrons=1024,
                               plasma_frequency=5,
                               dt=0.2 / 5,
                               NT=300 * 5
                               ),
        two_stream_instability("data_analysis/TS3/TS3.hdf5",
                               NG=64,
                               N_electrons=1024,
                               plasma_frequency=10,
                               dt=0.2 / 5,
                               NT=300 * 5
                               ),
        two_stream_instability("data_analysis/TSRANDOM1/TSRANDOM1.hdf5",
                               NG=64,
                               N_electrons=1024,
                               vrandom=1e-1,
                               ),
        two_stream_instability("data_analysis/TSRANDOM2/TSRANDOM2.hdf5",
                               NG=64,
                               N_electrons=1024,
                               plasma_frequency=5,
                               dt=0.2 / 5,
                               NT=300 * 5,
                               vrandom=1e-1,
                               ),
    ]

    show = True
    save = False
    for s in simulations:
        plotting.plotting(s, show=show, alpha=0.5, save=save)
