""" Run two stream instability"""
# coding=utf-8
import numpy as np

from pythonpic.algorithms import helper_functions
from pythonpic.algorithms.helper_functions import plotting_parser, Constants
from pythonpic.classes.grid import Grid
from pythonpic.classes.simulation import Simulation
from pythonpic.classes.species import Species
from pythonpic.visualization import plotting


def stability_condition(k0, v0, w0):
    dimensionless_number = k0 * v0 / w0
    expected_stability = dimensionless_number > 2 ** -0.5
    print(f"k0*v0/w0 is {dimensionless_number} which means the regime is "
          f"{'stable' if expected_stability else 'unstable'}")
    return expected_stability


def two_stream_instability(filename,
                           plasma_frequency=1,
                           qmratio=-1,
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
    grid = Grid(T=T, L=L, NG=NG, epsilon_0=epsilon_0)

    helper_functions.check_pusher_stability(plasma_frequency, grid.dt)
    np.random.seed(0)

    particle_mass = 1
    particle_charge = particle_mass * qmratio
    scaling = abs(particle_mass * plasma_frequency ** 2 * L / float(
        particle_charge * N_electrons * epsilon_0))

    filename = f"data_analysis/TS/{filename}/{filename}.hdf5"

    helper_functions.check_plasma_parameter(N_electrons * scaling, NG, grid.dx)
    k0 = 2 * np.pi / L

    expected_stability = stability_condition(k0, v0, plasma_frequency)

    electrons1 = Species(particle_charge, particle_mass, N_electrons, grid, "beam1", scaling=scaling)
    electrons2 = Species(species_2_sign * particle_charge, particle_mass, N_electrons, grid, "beam2", scaling=scaling)
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
    run = Simulation(grid.NT, grid.dt, list_species, grid, Constants(1, epsilon_0), filename=filename, title=description)
    # REFACTOR: add initial condition values to Simulation object
    run.grid_species_initialization()
    run.run(save_data)
    return run


def main():
    show, save, animate = plotting_parser("Two stream instability")
    simulations = [
        plotting.plots(two_stream_instability("TS1",
                                              NG=64,
                                              N_electrons=512,
                                              plasma_frequency=0.05 / 4,
                                              ), show=show, alpha=0.5, save=save, animate=animate),
        plotting.plots(two_stream_instability("TS2", NG=64, N_electrons=512, T=300 * 3 * 0.2),
                       show=show, alpha=0.5, save=save, animate=animate),
        plotting.plots(two_stream_instability("TS3", NG=64, N_electrons=1024, T=300 * 3 * 0.2),
                       show=show, alpha=0.5, save=save, animate=animate),
        plotting.plots(two_stream_instability("TSRANDOM1",
                                              NG=64,
                                              N_electrons=1024,
                                              vrandom=1e-1,
                                              ), show=show, alpha=0.5, save=save, animate=animate),
        plotting.plots(two_stream_instability("TSRANDOM2", NG=64, N_electrons=1024, T=300 * 5 * 0.2,
                                              vrandom=1e-1), show=show, alpha=0.5, save=save, animate=animate),
        ]

    for s in simulations:
        plotting.plots(s, show=show, alpha=0.5, save=save, animate=animate)


if __name__ == '__main__':
    main()
