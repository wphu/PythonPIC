import time
import argparse
import numpy as np
import FourierSolver
import Simulation
import diagnostics
from constants import epsilon_0
from gather import interpolateField
from scatter import charge_density_deposition
from parameters import NT, NG, N, T, dt, particle_mass, particle_charge, L, x_particles, v_particles, push_amplitude, push_mode
from Grid import Grid
from Species import Species

def field_quantities(x, charge_density):
    """ calculates field quantities"""
    # TODO: this is neither elegant nor efficient :( can probably be rewritten)
    dx = x[1] - x[0]
    potential, electric_field, fourier_field_energy = FourierSolver.PoissonSolver(charge_density, x)
    electric_field_function = lambda x_particles: interpolateField(x_particles, electric_field, x, dx)
    return potential, electric_field, electric_field_function, fourier_field_energy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="hdf5 file name for storing data")
    args = parser.parse_args()
    if args.filename[-5:] != ".hdf5":
        args.filename = args.filename + ".hdf5"

    g = Grid(L=2 * np.pi, NG=32)

    electrons = Species(particle_charge, particle_mass, N)
    electrons.distribute_uniformly(g.L)
    electrons.sinusoidal_position_perturbation(push_amplitude, push_mode, g.L)

    S = Simulation.Simulation(NT, g.NG, electrons.N, T, electrons.q, electrons.m, g.L, epsilon_0)

    g.gather_charge(electrons.x, electrons.q)
    fourier_field_energy = g.solve_poisson()

    # potential, electric_field, electric_field_function, fourier_field_energy = field_quantities(x, charge_density)
    kinetic_energy = electrons.leapfrog_init(g.electric_field_function, dt)

    start_time = time.time()
    for i in range(NT):
        S.update_grid(i, g.charge_density, g.electric_field)
        S.update_particles(i, electrons.x, electrons.v)

        kinetic, field, total = diagnostics.energies(x_particles, v_particles, particle_mass, g.dx, g.potential, g.charge_density)
        print("i{:4d} T{:12.3e} V{:12.3e} E{:12.3e}".format(i, kinetic, field, total))
        kinetic_energy = electrons.push_particles(g.electric_field_function, dt, g.L)
        g.gather_charge(electrons.x, electrons.q)
        fourier_field_energy = g.solve_poisson()
        kinetic = kinetic_energy.sum()
        diag = kinetic, fourier_field_energy, kinetic + fourier_field_energy
        S.update_diagnostics(i, diag)

    runtime = time.time() - start_time
    print("Runtime: {}".format(runtime))
    S.save_data(filename=args.filename)
