from collections import namedtuple

from numpy import pi

from Constants import Constants
from Grid import Grid
from Simulation import Simulation
from Species import Species
from helper_functions import date_version_string

species_args = namedtuple('species_arguments', ['N', 'q', 'm', 'initial_position', 'NT'])


class Runner():
    def __init__(self, NT: int = 1, dt: float = 0.1, epsilon_0: float = 1, c: float = 1,
                 NG: int = 32, L: float = 2 * pi,
                 *args, **kwargs):
        print(kwargs)
        self.NT = NT
        self.dt = dt
        self.constants = Constants(c, epsilon_0)
        self.grid = Grid(NG=NG, L=L, NT=NT)
        self.list_species = [Species(s.N, s.q, s.m) for name, s in kwargs.items()]
        self.simulation = Simulation(NT, dt, self.constants, self.grid, self.list_species)
        self.run_date = date_version_string()

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
    runner = Runner(species1=species_args(10, 1, 1, 'test', 10), )
    print(runner)
