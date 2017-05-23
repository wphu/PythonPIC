"""Implements interaction of the laser with a hydrogen shield plasma"""
# coding=utf-8
import numpy as np

from pythonpic.algorithms import helper_functions, BoundaryCondition
from pythonpic.algorithms.helper_functions import plotting_parser, Constants
from pythonpic.classes.grid import Grid
from pythonpic.classes.simulation import Simulation, load_data
from pythonpic.classes.species import Species
from pythonpic.visualization import plotting

laser_wavelength = 1.064e-6 # meters
laser_intensity = 1e23 # watt/meters squared
impulse_duration = 1e-13 # seconds

length = 1.0655e-5 # meters
total_time = 2e-13 # seconds
spatial_step = 7.7325e-9 # meters
number_cells = 1378

moat_length_left_side = 3.093e-6 # meters
# linear preplasma
preplasma_length = 7.73e-7 # meters
main_plasma_length = 7.73e-7 + preplasma_length # meters

electron_rest_mass = 9.109e-31 # kg
epsilon_zero = 8.854e-12 # F/m
electric_charge = 1.602e-19 # C
lightspeed = 2.998e8 #m /s
proton_mass = 1.6726219e-27 #kg

def critical_density(wavelength):
    """
    Calculates the critical plasma density:
    .. math::
    n_c = m_e \varepsilon_0 * (\frac{2 \pi c}{e \lambda})^2
    
    Parameters
    ----------
    wavelength : in meters

    Returns
    -------

    """
    n_c = electron_rest_mass * epsilon_zero * ((2 * np.pi * lightspeed ) / (electric_charge * wavelength))**2
    return n_c

maximum_electron_concentration = 5 * critical_density(laser_wavelength) # m^-3

# assert np.isclose(maximum_electron_concentration, 5.24e27), maximum_electron_concentration # m^-3
maximum_electron_concentration = 5.24e27 # TODO: this is a crutch

npic = 1.048e25 # m^-3

# n_macroparticles = 75000
n_macroparticles = 20000

scaling = npic # TODO: what should be the proper value here?

def laser(filename):
    filename=f"data_analysis/laser-shield/{filename}/{filename}.hdf5"
    dt = spatial_step / lightspeed
    grid = Grid(T=total_time, L=length, NG=number_cells, c = lightspeed, epsilon_0 = epsilon_zero)

    electrons = Species(-electric_charge, electron_rest_mass, n_macroparticles, grid, "electrons", scaling)
    protons = Species(electric_charge, proton_mass, n_macroparticles, grid, "protons", scaling)
    list_species = [electrons, protons]

    # bc = BoundaryCondition.non_periodic_bc(BoundaryCondition.Laser(laser_wavelength, dt*100, impulse_duration, c=lightspeed).laser_pulse)

    for species in list_species:
        print(f"Distributing {species.name} nonuniformly.")
        species.distribute_nonuniformly(length, moat_length_left_side, preplasma_length, main_plasma_length)
        species.random_position_perturbation(grid.L / grid.NG / 1000)

    description = "The big one"

    run = Simulation(grid.NT, dt, list_species, grid, Constants(lightspeed, epsilon_zero), filename=filename, title=description)
    print("Simulation prepared.")
    run.grid_species_initialization()
    print("Grid\species interactions initialized."
          "May Guod have mercy upon your soul."
          "Beginning simulation.")
    run.run(save_data=True, verbose=True)
    print("Well, that's it, then.")
    return run

def main():
    run = True
    if run:
        s = laser("Laser1")
    else:
        filename = "Laser1"
        filename=f"data_analysis/laser-shield/{filename}/{filename}.hdf5"
        s = load_data(filename)
    plotting.plots(s, show=True, animate=True)

if __name__ == '__main__':
    main()
