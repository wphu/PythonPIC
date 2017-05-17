""" Run two stream instability"""
# coding=utf-8
import numpy as np

import helper_functions
import plotting
from Grid import Grid
from Simulation import Simulation
from Species import Species
from helper_functions import plotting_parser, Constants


def stability_condition(k0, v0, w0):
    dimensionless_number = k0 * v0 / w0
    expected_stability = dimensionless_number > 2 ** -0.5
    print(f"k0*v0/w0 is {dimensionless_number} which means the regime is "
          f"{'stable' if expected_stability else 'unstable'}")
    return expected_stability


def two_stream_instability(filename,
                           plasma_frequency=1,
                           qmratio=-1,
                           dt=0.02,
                           T=300 * 0.2,
                           NG=32,
                           N_electrons=128,
                           L=2 * np.pi,
                           epsilon_0=1,
                           push_amplitude=0.001,
                           push_mode=1,
                           v0=0.05,
                           vrandom=0,
                           save_data: bool = True,
                           species_2_sign=1):
    """Implements two stream instability from Birdsall and Langdon"""
    print("Running two stream instability")

    helper_functions.check_pusher_stability(plasma_frequency, dt)
    NT = helper_functions.calculate_number_timesteps(T, dt)
    print(f"{NT} iterations to go.")
    np.random.seed(0)

    particle_mass = 1
    particle_charge = particle_mass * qmratio
    scaling = abs(particle_mass * plasma_frequency ** 2 * L / float(
        particle_charge * N_electrons * epsilon_0))

    filename = f"data_analysis/TS/{filename}/{filename}.hdf5"

    grid = Grid(L=L, NG=NG, NT=NT, n_species=2)
    helper_functions.check_plasma_parameter(N_electrons * scaling, NG, grid.dx)
    k0 = 2 * np.pi / L

    expected_stability = stability_condition(k0, v0, plasma_frequency)

    electrons1 = Species(particle_charge, particle_mass, N_electrons, "beam1", NT, scaling)
    electrons2 = Species(species_2_sign * particle_charge, particle_mass, N_electrons, "beam2", NT, scaling)
    electrons1.v[:, 0] = v0
    electrons2.v[:, 0] = -v0
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
    run = Simulation(NT, dt, list_species, grid, Constants(1, epsilon_0), filename=filename, title=description)
    # REFACTOR: add initial condition values to Simulation object
    run.grid_species_initialization()
    run.run(save_data)
    return run


def main():
    show, save, animate = plotting_parser("Two stream instability")
    simulations = [
        plotting.plotting(two_stream_instability("TS1",
                                                 NG=64,
                                                 N_electrons=512,
                                                 plasma_frequency=0.05 / 4,
                                                 ), show=show, alpha=0.5, save=save, animate=animate),
        plotting.plotting(two_stream_instability("TS2", NG=64, N_electrons=512, dt=0.03 / 5, T=300 * 3 * 0.2),
                          show=show, alpha=0.5, save=save, animate=animate),
        plotting.plotting(two_stream_instability("TS3", NG=64, N_electrons=1024, dt=0.01 / 3, T=300 * 3 * 0.2),
                          show=show, alpha=0.5, save=save, animate=animate),
        plotting.plotting(two_stream_instability("TSRANDOM1",
                                                 NG=64,
                                                 N_electrons=1024,
                                                 vrandom=1e-1,
                                                 ), show=show, alpha=0.5, save=save, animate=animate),
        plotting.plotting(two_stream_instability("TSRANDOM2", NG=64, N_electrons=1024, dt=0.2 / 5, T=300 * 5 * 0.2,
                                                 vrandom=1e-1), show=show, alpha=0.5, save=save, animate=animate),
        ]

    for s in simulations:
        plotting.plotting(s, show=show, alpha=0.5, save=save, animate=animate)


if __name__ == '__main__':
    main()
