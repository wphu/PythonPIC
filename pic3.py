import time
import argparse
import numpy as np
import Simulation
from parameters import NT, N, dt, push_amplitude, push_mode, epsilon_0
from Grid import Grid
from Species import Species
from helper_functions import date_version_string


def run(g, list_species, params, filename):
    NT, dt, epsilon_0 = params
    S = Simulation.Simulation(NT, dt, epsilon_0, g, list_species, date_version_string())
    g.gather_charge(list_species)
    fourier_field_energy = g.solve_poisson()
    for species in list_species:
        kinetic_energy = species.leapfrog_init(g.electric_field_function, dt)

    start_time = time.time()
    for i in range(NT):
        S.update_grid(i, g)
        S.update_particles(i, list_species)

        total_kinetic_energy = 0
        for species in list_species:
            kinetic_energy = species.push_particles(g.electric_field_function, dt, g.L).sum()
            S.kinetic_energy_history[species.name][i] = kinetic_energy
            total_kinetic_energy += kinetic_energy

        g.gather_charge(list_species)
        fourier_field_energy = g.solve_poisson()
        S.field_energy[i] = fourier_field_energy
        total_energy = fourier_field_energy + total_kinetic_energy
        S.total_energy[i] = total_energy
        print("i{:4d} T{:12.3e} V{:12.3e} E{:12.3e}".format(i, total_kinetic_energy, fourier_field_energy, total_energy))

    runtime = time.time() - start_time
    print("Runtime: {}".format(runtime))
    S.save_data(filename=args.filename)


def cold_plasma_oscillations(filename, dt, electron_charge, electron_mass, NT=150, NG=32, N_electrons=128, L=2 * np.pi, epsilon_0=1):
    g = Grid(L=L, NG=NG)
    electrons = Species(electron_charge, electron_mass, N_electrons, "electrons")
    list_species = [electrons]
    for species in list_species:
        species.distribute_uniformly(g.L)
        species.sinusoidal_position_perturbation(push_amplitude, push_mode, g.L)
    params = NT, dt, epsilon_0
    run(g, list_species, params, args.filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="hdf5 file name for storing data")
    args = parser.parse_args()
    if args.filename[-5:] != ".hdf5":
        args.filename = args.filename + ".hdf5"

    cold_plasma_oscillations(args.filename, dt, -1, 1)
