"""Implements interaction of the laser with a hydrogen shield plasma"""
# coding=utf-8
from pythonpic.algorithms import BoundaryCondition, FieldSolver
from pythonpic.algorithms.helper_functions import epsilon_zero, electric_charge, lightspeed, proton_mass, electron_rest_mass
from pythonpic.algorithms.helper_functions import plotting_parser, critical_density
from pythonpic.classes.grid import Grid
from pythonpic.classes.simulation import Simulation, load_simulation
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

maximum_electron_concentration = 5 * critical_density(laser_wavelength) # m^-3

# assert np.isclose(maximum_electron_concentration, 5.24e27), maximum_electron_concentration # m^-3
maximum_electron_concentration = 5.24e27 # CHECK: this is a crutch

npic = 1.048e25 # m^-3

n_macroparticles = 75000

scaling = npic # CHECK what should be the proper value here?

def laser(filename):
    filename=f"data_analysis/laser-shield/{filename}/{filename}.hdf5"
    dt = spatial_step / lightspeed
    bc = BoundaryCondition.Laser(laser_intensity, laser_wavelength, total_time/2, impulse_duration, c=lightspeed, epsilon_0=epsilon_zero).laser_pulse
    grid = Grid(T=total_time, L=length, NG=number_cells, c =lightspeed, epsilon_0 =epsilon_zero, bc=bc, periodic=False)

    electrons = Species(-electric_charge, electron_rest_mass, n_macroparticles, grid, "electrons", scaling)
    protons = Species(electric_charge, proton_mass, n_macroparticles, grid, "protons", scaling)
    list_species = [electrons, protons]


    for species in list_species:
        print(f"Distributing {species.name} nonuniformly.")
        species.distribute_nonuniformly(length, moat_length_left_side, preplasma_length, main_plasma_length)
        species.random_position_perturbation(grid.L / grid.NG / 1000)

    description = "The big one"

    run = Simulation(grid, list_species, filename=filename, title=description)
    print("Simulation prepared.")
    run.grid_species_initialization()
    print("Grid\species interactions initialized."
          "Beginning simulation.")
    run.run(save_data=True, verbose=True)
    print("Well, that's it, then.")
    return run

def main():
    import os
    args = plotting_parser("Hydrogen shield")
    filename = "Laser2"
    filename=f"data_analysis/laser-shield/{filename}/{filename}.hdf5"
    print(os.path.isfile(filename))
    run = False
    if run:
        s = laser("Laser2")
    else:
        s = load_simulation(filename)
    plotting.plots(s, *args)

if __name__ == '__main__':
    main()
