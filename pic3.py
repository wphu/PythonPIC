# coding=utf-8
""" A particle in cell code implemented in Python, with a focus on efficiency and optimization """
import argparse
import time

import numpy as np

import Simulation
from helper_functions import date_version_string


def run_electromagnetic(g, list_species, params, filename):
    """Full simulation run, with data gathering and saving to hdf5 file"""
    NT, dt, epsilon_0, B = params
    S = Simulation.Simulation(NT, dt, epsilon_0, g, list_species, date_version_string())
    g.gather_charge(list_species)
    g.solve_poisson()

    def magnetic_field_function(x):
        result = np.zeros((x.size, 3))
        result[:, 2] = B
        return result

    for species in list_species:
        species.boris_init(g.electric_field_function, magnetic_field_function, dt, g.L)

    start_time = time.time()
    for i in range(NT):
        g.save_field_values(i)
        total_kinetic_energy = 0

        for species in list_species:
            species.save_particle_values(i)
            # 1. GATHER FIELD TO PARTICLES
            # 2. INTEGRATE EQUATIONS OF MOTION
            kinetic_energy = species.boris_push_particles(g.electric_field_function,
                                                          magnetic_field_function, dt, g.L).sum()
            # TODO: remove sum from this place
            species.kinetic_energy_history[i] = kinetic_energy
            total_kinetic_energy += kinetic_energy

        # 2. SCATTER CHARGE AND CURRENT TO GRID
        g.gather_charge(list_species)
        g.gather_current(list_species)

        fourier_field_energy = g.solve_poisson()
        g.iterate_EM_field()

        g.grid_energy_history[i] = fourier_field_energy
        total_energy = fourier_field_energy + total_kinetic_energy
        S.total_energy[i] = total_energy

    runtime = time.time() - start_time
    print("Runtime: {}".format(runtime))

    if filename[-5:] != ".hdf5":
        filename = args.filename + ".hdf5"
    S.save_data(filename=filename, runtime=runtime)


if __name__ == "__main__":
    from configs.run_twostream import two_stream_instability

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="hdf5 file name for storing data")
    args = parser.parse_args()
    if args.filename[-5:] != ".hdf5":
        args.filename += ".hdf5"

    two_stream_instability("data_analysis/" + args.filename)
