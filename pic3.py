import time
import argparse
import numpy as np
import Simulation
from parameters import NT, N, dt, push_amplitude, push_mode, epsilon_0
from Grid import Grid
from Species import Species
from helper_functions import date_version_string

def run(g, list_species, params, filename):
    S = Simulation.Simulation(NT, dt, epsilon_0, g, list_species, date_version_string())
    for species in list_species:
        g.gather_charge(species)
    fourier_field_energy = g.solve_poisson()
    for species in list_species:
        kinetic_energy = species.leapfrog_init(g.electric_field_function, dt)

    start_time = time.time()
    for i in range(NT):
        S.update_grid(i, g)
        S.update_particles(i, list_species)

        total_kinetic_energy = 0
        for species in list_species:
            kinetic_energy = species.push_particles(g.electric_field_function, dt, g.L)
            g.gather_charge(species)
            total_kinetic_energy = kinetic_energy.sum()
        fourier_field_energy = g.solve_poisson()
        diag = total_kinetic_energy, fourier_field_energy, total_kinetic_energy + fourier_field_energy
        S.update_diagnostics(i, *diag)
        print("i{:4d} T{:12.3e} V{:12.3e} E{:12.3e}".format(i, *diag))

    runtime = time.time() - start_time
    print("Runtime: {}".format(runtime))
    S.save_data(filename=args.filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="hdf5 file name for storing data")
    args = parser.parse_args()
    if args.filename[-5:] != ".hdf5":
        args.filename = args.filename + ".hdf5"

    g = Grid(L=2 * np.pi, NG=32)
    electrons = Species(-1.0, 1.0, N)
    electrons.distribute_uniformly(g.L)
    electrons.sinusoidal_position_perturbation(push_amplitude, push_mode, g.L)
    list_species = [electrons]
    params = NT, dt, epsilon_0

    run(g, list_species, params, args.filename)
