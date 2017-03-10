"""Class representing a run of the simulation"""
# coding=utf-8
from enum import Enum

from numpy import pi

from Species import Species

initial_positions = Enum('initial positions', ['uniform', "position_perturbation"])


class Runner:
    """
    Represents a simulation run.
    """

    def __init__(self, NT: int = 1, dt: float = 0.1, epsilon_0: float = 1, c: float = 1, NG: int = 32,
                 L: float = 2 * pi, T=None, **list_species: dict):
        """
        Initial conditions and settings for the simulation
        :param int NT: number of iterations
        :param float dt: iteration time step
        :param float epsilon_0: the physical constant
        :param float c: the speed of light
        :param int NG: number of grid points
        :param float L: grid length
        :param str filename: path to hdf5 file, should end in .hdf5. default is current date
        :param args: positional arguments, currently unused
        :param species_args list_species: settings for particle species
        """
        self.list_species = []
        for name, arguments in list_species.items():
            if type(arguments) is dict:
                s = Species(arguments['q'], arguments['m'], arguments['N'], arguments['name'], arguments['NT'])
                if arguments['initial_position'] == initial_positions.uniform.name:
                    s.distribute_uniformly(self.grid.L)
                elif arguments['initial_position'] == initial_positions.position_perturbation.name:
                    s.distribute_uniformly(self.grid.L)
                    s.sinusoidal_position_perturbation(arguments['mode_amplitude'], arguments['mode_number'],
                                                       self.grid.L)
                # TODO: if arguments.initial_position == initial_positions.sinusoidal
                particles_in_grid = s.x.max() < self.grid.L and s.x.min() >= 0
                assert particles_in_grid
                self.list_species.append(s)



