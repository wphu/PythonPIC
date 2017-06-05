"""Implements interaction of the laser with a hydrogen shield plasma"""
# coding=utf-8
from ..algorithms import BoundaryCondition
from ..classes import Grid, Simulation, Species
from ..helper_functions.physics import epsilon_zero, electric_charge, lightspeed, proton_mass, electron_rest_mass, \
    critical_density

from functools import partial
from ..visualization.plotting import plots
from ..visualization import animation
plots = partial(plots, animation_type = animation.FullAnimation)

VERSION = 11
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

print("crit density", critical_density(laser_wavelength))
maximum_electron_concentration = 5 * critical_density(laser_wavelength) # m^-3

# assert np.isclose(maximum_electron_concentration, 5.24e27), maximum_electron_concentration # m^-3
maximum_electron_concentration = 5.24e27 # CHECK: this is a crutch

npic = 1.048e25 # m^-3

N_MACROPARTICLES = 75000
n_macroparticles = N_MACROPARTICLES
scaling = npic # CHECK what should be the proper value here?
default_scaling = npic # CHECK what should be the proper value here?

category_name = "laser-shield"
class laser(Simulation):
    def __init__(self, filename, n_macroparticles, impulse_duration, laser_intensity, perturbation_amplitude):
        """
        A simulation of laser-hydrogen shield interaction.

        Parameters
        ----------
        filename : str
            Filename for the simulation.
        n_macroparticles : int
            Number of macroparticles for each species. The simulation is
            normalized to 75000 macroparticles by default,
        impulse_duration : float
            Duration of the laser impulse.
        laser_intensity : float
            Laser impulse intensity, in W/m^2. A good default is 1e21.
        perturbation_amplitude : float
            Amplitude of the initial position perturbation.
        """
        if laser_intensity:
            bc_laser = BoundaryCondition.Laser(laser_intensity=laser_intensity,
                                         laser_wavelength=laser_wavelength,
                                         envelope_center_t = total_time/2,
                                         envelope_width=impulse_duration,
                                         envelope_power=6,
                                         c=lightspeed,
                                         epsilon_0=epsilon_zero,
                                         )
            print(f"Laser amplitude: {bc_laser.laser_amplitude:e}")
            bc = bc_laser.laser_pulse
        else:
            bc = lambda x: None
        grid = Grid(T=total_time, L=length, NG=number_cells, c =lightspeed, epsilon_0 =epsilon_zero, bc=bc, periodic=False)

        if n_macroparticles:
            scaling = default_scaling * N_MACROPARTICLES / n_macroparticles
            electrons = Species(-electric_charge, electron_rest_mass, n_macroparticles, grid, "electrons", scaling)
            protons = Species(electric_charge, proton_mass, n_macroparticles, grid, "protons", scaling)
            list_species = [electrons, protons]
        else:
            list_species = []

        self.perturbation_amplitude = perturbation_amplitude # in units of dx

        description = "Hydrogen shield-laser interaction"

        super().__init__(grid, list_species,
                         filename=filename,
                         category_type="laser-shield",
                         config_version=VERSION,
                         title=description,
                         considered_large = True)
        print("Simulation prepared.")

    def grid_species_initialization(self):
        for species in self.list_species:
            print(f"Distributing {species.name} nonuniformly.")
            species.distribute_nonuniformly(length, moat_length_left_side, preplasma_length, main_plasma_length)
            species.random_position_perturbation(self.perturbation_amplitude)
        print("Finished initial distribution of particles.")
        super().grid_species_initialization()
        print("Finished initialization.")


