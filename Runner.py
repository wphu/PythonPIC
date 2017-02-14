import time
from collections import namedtuple
from enum import Enum

from numpy import pi

from Constants import Constants
from Grid import Grid
from Simulation import Simulation
from Species import Species
from helper_functions import date_version_string

species_args = namedtuple('species_arguments', ['N', 'q', 'm', 'NT', 'name', 'initial_position'])
initial_positions = Enum('initial positions', 'uniform')

class Runner():
    """
    Represents a simulation run.
    """
    def __init__(self, NT: int = 1, dt: float = 0.1, epsilon_0: float = 1, c: float = 1,
                 NG: int = 32, L: float = 2 * pi,
                 filename=time.strftime("%Y-%m-%d_%H-%M-%S.hdf5"),  # TODO: this is probably a default
                 *args, **kwargs):
        """
        Initial conditions and settings for the simulation
        :param int NT: number of iterations
        :param float dt: iteration timestep
        :param float epsilon_0: the physical constant
        :param float c: the speed of light
        :param int NG: number of grid points
        :param float L: grid length
        :param str filename: path to hdf5 file, should end in .hdf5. default is current date
        :param args: positional arguments, currently unused
        :param species_args kwargs: settings for particle species
        """
        self.NT = NT
        self.dt = dt  # TODO: allow passing total simulation time
        self.constants = Constants(c, epsilon_0)
        self.grid = Grid(NG=NG, L=L, NT=NT)
        self.list_species = []
        for name, arguments in kwargs.items():
            if type(arguments) is species_args:
                s = Species(arguments.q, arguments.m, arguments.N, arguments.name, arguments.NT)
                if arguments.initial_position == initial_positions.uniform:
                    s.distribute_uniformly(self.grid.L)
                # TODO: if arguments.initial_position == initial_positions.sinusoidal
                particles_in_grid = s.x.max() < self.grid.L and s.x.min() >= 0
                assert particles_in_grid
                self.list_species.append(s)

        self.simulation = Simulation(NT, dt, self.constants, self.grid, self.list_species)
        self.run_date = date_version_string()
        self.filename = filename

    def grid_species_initialization(self):
        """
        Initializes grid and particle relations:
        1. gathers charge from particles to grid
        2. solves Poisson equation to get initial field
        3. initializes pusher via a step back
        """
        self.grid.gather_charge(self.list_species)
        self.grid.solve_poisson()  # TODO: abstract field solver
        for species in self.list_species:
            # TODO: abstract pusher
            species.leapfrog_init(self.grid.electric_field_function, self.dt)

    def iteration(self, i: int):
        """

        :param int i: iteration number
        Runs an iteration step
        1. saves field values
        2. for all particles:
            2. 1. saves particle values
            2. 2. pushes particles forward

        """
        self.grid.save_field_values(i)  # TODO: is this necessary with what happens after loop

        total_kinetic_energy = 0  # accumulate over species
        for species in self.list_species:
            species.save_particle_values(i)
            kinetic_energy = species.leapfrog_push(self.grid.electric_field_function,
                                                   self.dt,
                                                   self.grid.L).sum()
            # TODO: remove this sum
            species.kinetic_energy_history[i] = kinetic_energy
            total_kinetic_energy += kinetic_energy

        self.grid.gather_charge(self.list_species)
        fourier_field_energy = self.grid.solve_poisson()
        self.grid.grid_energy_history[i] = fourier_field_energy
        self.simulation.total_energy[i] = total_kinetic_energy + fourier_field_energy

    def run(self, n: int = -1, save_data=True):
        """
        Run n iterations of the simulation, timing it.
        :param int n: how many iterations to run (starting from 0!)
                               (default is self.NT)
        :return float runtime: runtime for this part of run, in seconds
        """
        if n == -1:
            n = self.NT
        start_time = time.time()
        for i in range(n):
            self.iteration(i)
            # TODO: run for X iterations, save data, run again? needs a static counter
        runtime = time.time() - start_time

        if save_data:
            self.simulation.save_data(filename=self.filename, runtime=runtime)
        return runtime

    def __str__(self, *args, **kwargs):
        result_string = f"""
        Runner from {self.run_date} containing:
        Epsilon zero = {self.constants.epsilon_0}, c = {self.constants.epsilon_0}
        {self.NT} iterations with timestep {self.dt}
        {self.grid.NG}-cell grid of length {self.grid.L:.2f}""".lstrip()
        for species in self.list_species:
            result_string = result_string + "\n\t" + str(species)
        return result_string


if __name__ == '__main__':
    runner = Runner(species1=species_args(1, 1, 1, 10, 'test particles', initial_positions.uniform))
    runner.grid_species_initialization()
    runner.run(save_data=False)
    print(runner)
    print("Run completed!")
