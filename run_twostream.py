""" Run two stream instability"""
# coding=utf-8
import numpy as np

import plotting
from Constants import Constants
from Grid import Grid
from Simulation import Simulation
from Species import Species
from helper_functions import plotting_parser


def two_stream_instability(filename,
                           plasma_frequency=1,
                           qmratio=-1,
                           dt=0.2,
                           NT=300,
                           NG=32,
                           N_electrons=128,
                           L=2 * np.pi,
                           epsilon_0=1,
                           push_amplitude=0.001,
                           push_mode=1,
                           v0=1.0,
                           vrandom=0,
                           save_data: bool = True,
                           species_2_sign=1):
    """Implements two stream instability from Birdsall and Langdon"""
    print("Running two stream instability")
    particle_mass = 1
    particle_charge = particle_mass * qmratio
    scaling = abs(particle_mass * plasma_frequency ** 2 * L / float(
        particle_charge * N_electrons * epsilon_0))

    filename = f"data_analysis/TS/{filename}/{filename}.hdf5"

    grid = Grid(L=L, NG=NG, NT=NT, n_species=2)
    k0 = 2 * np.pi / L
    w0 = plasma_frequency
    expected_stability = k0 * v0 / w0 > 2 ** -0.5
    print("k0*v0/w0 is", k0 * v0 / w0, "which means the regime is", "stable" if expected_stability else "unstable")
    electrons1 = Species(particle_charge, particle_mass, N_electrons, "beam1", NT, scaling)
    electrons2 = Species(species_2_sign * particle_charge, particle_mass, N_electrons, "beam2", NT, scaling)
    electrons1.v[:] = v0
    electrons2.v[:] = -v0
    list_species = [electrons1, electrons2]
    for i, species in enumerate(list_species):
        species.distribute_uniformly(L, 0.5 * grid.dx * i)
        species.sinusoidal_position_perturbation(push_amplitude, push_mode, grid.L)
        if vrandom:
            species.random_velocity_perturbation(0, vrandom)
    description = f"Two stream instability - two beams counterstreaming with $v_0$ {v0:.2f}"
    if vrandom:
        description += f" + thermal $v_1$ of standard dev. {vrandom:.2f}"

    description += f" ({'stable' if expected_stability else 'unstable'}).\n"
    run = Simulation(NT, dt, Constants(1, epsilon_0),
                     grid, list_species, filename=filename, title=description)
    # REFACTOR: add initial condition values to Simulation object
    run.grid_species_initialization()
    run.run(save_data)
    return run


if __name__ == '__main__':
    np.random.seed(0)
    simulations = [
        two_stream_instability("TS1",
                               NG=64,
                               N_electrons=512,
                               plasma_frequency=5,
                               dt=0.2 / 5,
                               NT=300 * 5
                               ),
        two_stream_instability("TS3",
                               NG=64,
                               N_electrons=1024,
                               plasma_frequency=10,
                               dt=0.2 / 5,
                               NT=300 * 5
                               ),
        two_stream_instability("TSRANDOM1",
                               NG=64,
                               N_electrons=1024,
                               vrandom=1e-1,
                               ),
        two_stream_instability("TSRANDOM2",
                               NG=64,
                               N_electrons=1024,
                               plasma_frequency=5,
                               dt=0.2 / 5,
                               NT=300 * 5,
                               vrandom=1e-1,
                               ),
        ]

    show, save, animate = plotting_parser("Two stream instability")
    for s in simulations:
        plotting.plotting(s, show=show, alpha=0.5, save=save, animate=animate)
