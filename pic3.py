""" A particle in cell code implemented in Python, with a focus on efficiency and optimization """
import time
import argparse
import numpy as np
import Simulation
from Grid import Grid
from Species import Species
from helper_functions import date_version_string


def run(g, list_species, params, filename):
    """Full simulation run, with data gathering and saving to hdf5 file"""
    NT, dt, epsilon_0 = params
    S = Simulation.Simulation(NT, dt, epsilon_0, g, list_species, date_version_string())
    g.gather_charge(list_species)
    fourier_field_energy = g.solve_poisson()
    for species in list_species:
        species.leapfrog_init(g.electric_field_function, dt)

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
        S.energy_per_mode[i] = g.energy_per_mode

    runtime = time.time() - start_time
    print("Runtime: {}".format(runtime))

    if filename[-5:] != ".hdf5":
        filename = args.filename + ".hdf5"
    S.save_data(filename=filename)

def cold_plasma_oscillations(filename, plasma_frequency=1, qmratio=-1, dt=0.2, NT=150,
                             NG=32, N_electrons=128, L=2 * np.pi, epsilon_0=1,
                             push_amplitude=0.001, push_mode=1):

    """Implements cold plasma oscillations from Birdsall and Langdon

    (plasma excited by a single cosinusoidal mode via position displacements)"""
    print("Running cold plasma oscillations")
    particle_charge = plasma_frequency**2 * L / float(N_electrons * epsilon_0 * qmratio)
    particle_mass = particle_charge / qmratio

    g = Grid(L=L, NG=NG)
    electrons = Species(particle_charge, particle_mass, N_electrons, "electrons")
    list_species = [electrons]
    for species in list_species:
        species.distribute_uniformly(g.L)
        species.sinusoidal_position_perturbation(push_amplitude, push_mode, g.L)
    params = NT, dt, epsilon_0
    return run(g, list_species, params, filename)

def two_stream_instability(filename, plasma_frequency=1, qmratio=-1, dt=0.2, NT=300,
                             NG=32, N_electrons=128, L=2 * np.pi, epsilon_0=1,
                             push_amplitude=0.001, push_mode=1, v0=1.0):
    """Implements two stream instability from Birdsall and Langdon"""
    print("Running two stream instability")
    particle_charge = plasma_frequency**2 * L / float(2*N_electrons * epsilon_0 * qmratio)
    particle_mass = particle_charge / qmratio

    g = Grid(L=L, NG=NG)
    k0 = 2*np.pi/g.L
    w0 = plasma_frequency
    print("k0*v0/w0 is", k0*v0/w0, "which means the regime is", "stable" if k0*v0/w0 < 2**0.5 else "unstable")
    electrons1 = Species(particle_charge, particle_mass, N_electrons, "beam1")
    electrons2 = Species(particle_charge, particle_mass, N_electrons, "beam2")
    electrons1.v[:] = v0
    electrons2.v[:] = -v0
    list_species = [electrons1, electrons2]
    for i, species in enumerate(list_species):
        species.distribute_uniformly(g.L, 0.5*g.dx*i)
        species.sinusoidal_position_perturbation(push_amplitude, push_mode, g.L)
    params = NT, dt, epsilon_0
    return run(g, list_species, params, filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="hdf5 file name for storing data")
    args = parser.parse_args()
    if args.filename[-5:] != ".hdf5":
        args.filename = args.filename + ".hdf5"

    two_stream_instability("data_analysis/" + args.filename)
