import time
import argparse
import numpy as np
import Simulation
from Grid import Grid
from Species import Species
from helper_functions import date_version_string


def run(g, list_species, params, filename):
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
        # print("i{:4d} T{:12.3e} V{:12.3e} E{:12.3e}".format(i, total_kinetic_energy, fourier_field_energy, total_energy))

    runtime = time.time() - start_time
    print("Runtime: {}".format(runtime))
    S.save_data(filename=filename)

    # return S.grid.k_plot, S.energy_per_mode.sum(axis=0)*S.dt/2

    # ratio = S.kinetic_energy_history['electrons'].max()/S.field_energy.max()
    # print("ratio ", ratio)
    # return ratio

def cold_plasma_oscillations(filename, plasma_frequency=1, qmratio=-1, dt=0.2, NT=150,
                             NG=32, N_electrons=128, L=2 * np.pi, epsilon_0=1,
                             push_amplitude=0.001, push_mode=1):

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
                             push_amplitude=0.001, push_mode=1, v1=1., v2=-1.):

    particle_charge = plasma_frequency**2 * L / float(2*N_electrons * epsilon_0 * qmratio)
    particle_mass = particle_charge / qmratio

    g = Grid(L=L, NG=NG)
    k0 = 2*np.pi/g.L
    w0 = plasma_frequency
    print(k0*v1/w0)
    electrons1 = Species(particle_charge, particle_mass, N_electrons, "beam1")
    electrons2 = Species(particle_charge, particle_mass, N_electrons, "beam2")
    electrons1.v[:] = v1
    electrons2.v[:] = v2
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

    cold_plasma_oscillations("1default.hdf5")
    cold_plasma_oscillations("2dense.hdf5", N_electrons=2**9+1)
    cold_plasma_oscillations("3leapfrog_instability_pre.hdf5", dt = 0.2, NT=150, N_electrons=128, NG=32)
    cold_plasma_oscillations("4leapfrog_instability_post.hdf5", dt = 3, NT=150, N_electrons=2**9+1, NG=32)
    cold_plasma_oscillations("5wavebreaking.hdf5", dt = 0.2, NT=150, N_electrons=2**11+1, NG=32, push_amplitude=2)
    cold_plasma_oscillations("6aliasing.hdf5", dt = 0.2, NT=150, N_electrons=2**9+1, NG=32, push_mode=18)

    two_stream_instability("ts1default.hdf5")
    two_stream_instability("ts2long.hdf5", NT=600)
    two_stream_instability("ts3growth.hdf5", N_electrons=4096, NT=600, v1=1, v2=-1, push_amplitude=0.01)
    two_stream_instability("ts4growth.hdf5", N_electrons=4096, NT=600, v1=2, v2=-2, push_amplitude=0.01)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(*res)
    # plt.hlines(0.9952, 0, 16)
    # plt.show()
    # ratios = [cold_plasma_oscillations(args.filename, dt = 0.02, NT=1500, L=l, N_electrons=512) for l in L]
    # print(ratios)
    # plt.figure()
    # plt.plot(L, ratios)
    # plt.show()
