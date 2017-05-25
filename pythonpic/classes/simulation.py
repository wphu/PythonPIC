"""Data interface class"""
# coding=utf-8
import os
import time

import h5py
import numpy as np

from ..algorithms import helper_functions, BoundaryCondition
from ..algorithms.helper_functions import git_version, Constants
from .grid import Grid
from .species import Species

class Simulation:
    """Contains data from one run of the simulation:
    Parameters
    ----------
    grid : Grid
    list_species : list
    run_date : str
    git_ver : str
    filename : str
    title : str
    """
    def __init__(self, grid: Grid, list_species, run_date=time.ctime(), git_ver=git_version(),
                 filename=time.strftime("%Y-%m-%d_%H-%M-%S.hdf5"), boundary_condition=BoundaryCondition.PeriodicBC, title=""):
        self.NT = grid.NT
        self.dt = grid.dt
        self.t = np.arange(self.NT) * self.dt
        self.grid = grid
        self.list_species = list_species
        self.field_energy = np.zeros(self.NT)
        self.total_energy = np.zeros(self.NT)
        self.boundary_condition = boundary_condition
        self.constants = Constants(grid.c, grid.epsilon_0)
        self.filename = filename
        self.title = title
        self.git_version = git_ver
        self.run_date = run_date

    def grid_species_initialization(self):
        """
        Initializes grid and particle relations:
        1. gathers charge from particles to grid
        2. solves Poisson equation to get initial field
        3. initializes pusher via a step back
        """
        self.grid.gather_charge(self.list_species)
        self.grid.gather_current(self.list_species)
        self.grid.init_solver()
        self.grid.apply_bc(0)
        for species in self.list_species:
            species.init_push(self.grid.electric_field_function, self.grid.magnetic_field_function)

    def iteration(self, i: int, periodic: bool = True):
        """

        :param periodic: is the simulation periodic? (affects boundary conditions)
        :type periodic: bool
        :param int i: iteration number
        Runs an iteration step
        1. saves field values
        2. for all particles:
            2. 1. saves particle values
            2. 2. pushes particles forward

        """
        self.grid.save_field_values(i)  # TODO: is this the right place, or after loop?

        total_kinetic_energy = 0  # accumulate over species
        for species in self.list_species:
            species.save_particle_values(i)
            total_kinetic_energy += species.push(self.grid.electric_field_function, self.grid.magnetic_field_function)
            species.apply_bc()
        self.grid.apply_bc(i)
        self.grid.gather_charge(self.list_species)
        self.grid.gather_current(self.list_species)
        fourier_field_energy = self.grid.solve()
        self.grid.grid_energy_history[i] = fourier_field_energy
        self.total_energy[i] = total_kinetic_energy + fourier_field_energy

    def run(self, save_data: bool = True, verbose = False) -> float:
        """
        Run n iterations of the simulation, saving data as it goes.
        Parameters
        ----------
        save_data (bool): Whether or not to save the data
        verbose (bool): Whether or not to print out progress

        Returns
        -------
        runtime (float): runtime of this part of simulation in seconds
        """
        start_time = time.time()
        for i in range(self.NT):
            if verbose and i % (self.NT // 100) == 0:
                print(f"{i}/{self.NT} iterations ({i/self.NT*100:.0f}%) done!")
            self.iteration(i)
        for species in self.list_species:
            species.save_particle_values(self.NT)
        runtime = time.time() - start_time
        if self.filename and save_data:
            self.save_data(filename=self.filename, runtime=runtime)
        return runtime

    def save_data(self, filename: str = time.strftime("%Y-%m-%d_%H-%M-%S.hdf5"), runtime: bool = False) -> str:
        """Save simulation data to hdf5.
        filename by default is the timestamp for the simulation."""
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with h5py.File(filename, "w") as f:
            grid_data = f.create_group('grid')
            self.grid.save_to_h5py(grid_data)

            all_species = f.create_group('species')
            for species in self.list_species:
                species_data = all_species.create_group(species.name)
                species.save_to_h5py(species_data)
            f.create_dataset(name="Field energy", dtype=float, data=self.field_energy)
            f.create_dataset(name="Total energy", dtype=float, data=self.total_energy)

            f.attrs['dt'] = self.dt
            f.attrs['NT'] = self.NT
            f.attrs['run_date'] = self.run_date
            f.attrs['git_version'] = self.git_version
            f.attrs['title'] = self.title
            if runtime:
                f.attrs['runtime'] = runtime
        print("Saved file to {}".format(filename))
        return filename

    def __str__(self, *args, **kwargs):
        result_string = f"""
        {self.title} simulation ({os.path.basename(self.filename)}) containing {self.NT} iterations with time step {
        self.dt:.3e}
        Done on {self.run_date} from git version {self.git_version}
        {self.grid.NG}-cell grid of length {self.grid.L:.2f}. Epsilon zero = {self.constants.epsilon_0}, 
        c = {self.constants.epsilon_0}""".lstrip()
        for species in self.list_species:
            result_string = result_string + "\n" + str(species)
        return result_string  # REFACTOR: add information from config file (run_coldplasma...)


# class PostprocessedSimulation # TODO
def load_data(filename: str) -> Simulation:
    """Create a Simulation object from a hdf5 file"""
    with h5py.File(filename, "r") as f:
        total_energy = f['Total energy'][...]

        NT = f.attrs['NT']
        dt = f.attrs['dt']
        title = f.attrs['title']

        grid_data = f['grid']
        NG = grid_data.attrs['NGrid']
        grid = Grid(L=NT, NG=NG)
        grid.load_from_h5py(grid_data)

        all_species = []
        for species_group_name in f['species']:
            species_group = f['species'][species_group_name]
            species = Species(1, 1, 1, NT=NT)
            species.load_from_h5py(species_group)
            all_species.append(species)
        run_date = f.attrs['run_date']
        git_version = f.attrs['git_version']
    S = Simulation(grid, all_species, git_version, filename=filename, title=title)

    S.total_energy = total_energy

    return S
