"""Implements interaction of the laser with a hydrogen shield plasma"""
# coding=utf-8
from ..algorithms import BoundaryCondition
from ..helper_functions.physics import epsilon_zero, electric_charge, lightspeed, proton_mass, electron_rest_mass, \
    critical_density
from ..helper_functions.helpers import plotting_parser
from ..classes import Grid, Simulation, Species
from ..visualization import plotting

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

category_name = "laser-shield"
def laser(filename, n_macroparticles, impulse_duration):
    bc = BoundaryCondition.Laser(laser_intensity=laser_intensity,
                                 laser_wavelength=laser_wavelength,
                                 envelope_center_t = total_time/2,
                                 envelope_width=impulse_duration,
                                 c=lightspeed,
                                 epsilon_0=epsilon_zero,
                                 ).laser_pulse
    grid = Grid(T=total_time, L=length, NG=number_cells, c =lightspeed, epsilon_0 =epsilon_zero, bc=bc, periodic=False)

    if n_macroparticles:
        electrons = Species(-electric_charge, electron_rest_mass, n_macroparticles, grid, "electrons", scaling)
        protons = Species(electric_charge, proton_mass, n_macroparticles, grid, "protons", scaling)
        list_species = [electrons, protons]


        for species in list_species:
            print(f"Distributing {species.name} nonuniformly.")
            species.distribute_nonuniformly(length, moat_length_left_side, preplasma_length, main_plasma_length)
            species.random_position_perturbation(grid.L / grid.NG / 1000)
    else:
        list_species = []

    description = "The big one"

    run = Simulation(grid, list_species,
                     filename=filename,
                     category_type="laser-shield",
                     config_version=4,
                     title=description)
    print("Simulation prepared.")
    return run


def main():
    args = plotting_parser("Hydrogen shield")
    s =laser("field_only", 0, impulse_duration).lazy_run()
    plotting.plots(s, *args)
    s =laser("few_particles", 10000, impulse_duration).lazy_run()
    plotting.plots(s, *args)
    s = laser("few_particles_short_pulse", 10000, impulse_duration/10).lazy_run()
    plotting.plots(s, *args)
    s = laser("production_run_short_pulse", n_macroparticles, impulse_duration/10).lazy_run()
    plotting.plots(s, *args)
    s = laser("production_run", n_macroparticles, impulse_duration).lazy_run()
    plotting.plots(s, *args)

if __name__ == '__main__':
    main()
