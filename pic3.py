""" A particle in cell code implemented in Python, with a focus on efficiency and optimization """
import time
import argparse
import Simulation
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
        # S.update_grid(i, g)
        g.save_field_values(i)
        # S.update_particles(i, list_species)

        total_kinetic_energy = 0
        for species in list_species:
            species.save_particle_values(i)
            kinetic_energy = species.push_particles(g.electric_field_function, dt, g.L).sum()
            species.kinetic_energy_history[i] = kinetic_energy
            total_kinetic_energy += kinetic_energy

        g.gather_charge(list_species)
        fourier_field_energy = g.solve_poisson()
        g.grid_energy_history[i] = fourier_field_energy
        # g.grid_energy_history[i] = fourier_field_energy
        total_energy = fourier_field_energy + total_kinetic_energy
        S.total_energy[i] = total_energy
        # import ipdb; ipdb.set_trace()
        # S.grid.energy_per_mode_history[i] = g.energy_per_mode

    runtime = time.time() - start_time
    print("Runtime: {}".format(runtime))

    if filename[-5:] != ".hdf5":
        filename = args.filename + ".hdf5"
    S.save_data(filename=filename)

if __name__ == "__main__":
    from run_twostream import two_stream_instability
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="hdf5 file name for storing data")
    args = parser.parse_args()
    if args.filename[-5:] != ".hdf5":
        args.filename = args.filename + ".hdf5"

    two_stream_instability("data_analysis/" + args.filename)
