"""Data interface class"""
# coding=utf-8
import os
import time

import h5py
import numpy as np

from .grid import Grid, load_grid
from .species import Species, load_species
from ..algorithms import BoundaryCondition
from ..algorithms.helper_functions import git_version, Constants, report_progress


class Simulation:
    """

    Contains data from one run of the simulation.

    Parameters
    ----------
    grid : Grid
    list_species : list
    run_date : str
    git_ver : str
    filename : str
    title : str
    """
    def __init__(self, grid: Grid, list_species=None, run_date=time.ctime(), git_version=git_version(),
                 filename=time.strftime("%Y-%m-%d_%H-%M-%S.hdf5"), title=""):
        self.NT = grid.NT
        self.dt = grid.dt
        self.t = np.arange(self.NT) * self.dt
        self.grid = grid
        if list_species is None:
            list_species = []
        self.list_species = list_species
        self.field_energy = np.zeros(self.NT)
        self.total_energy = np.zeros(self.NT)
        if not filename.endswith(".hdf5"):
            raise ValueError("Filename does not end with '.hdf5'.")
        self.filename = filename
        self.title = title
        self.git_version = git_version
        self.run_date = run_date

        self.postprocessed=False

    def postprocess(self):
        if not self.postprocessed:
            self.grid.postprocess()
            for species in self.list_species:
                species.postprocess()
            self.postprocessed = True

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
        self.grid.save_field_values(i)  # CHECK: is this the right place, or after loop?

        total_kinetic_energy = 0  # accumulate over species
        for species in self.list_species:
            species.save_particle_values(i)
            total_kinetic_energy += species.push(self.grid.electric_field_function, self.grid.magnetic_field_function)
            species.apply_bc()
        self.grid.apply_bc(i)
        self.grid.gather_charge(self.list_species)
        self.grid.gather_current(self.list_species)
        fourier_field_energy = self.grid.solve()
        # self.grid.grid_energy_history[i] = fourier_field_energy # TODO: readd
        # self.total_energy[i] = total_kinetic_energy + fourier_field_energy # TODO: readd

    def run(self, save_data: bool = True, postprocess=True, verbose = False) -> float:
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
                report_progress(i, self.NT)
            self.iteration(i)
        # for species in self.list_species:
        #     species.save_particle_values(self.NT)
        runtime = time.time() - start_time
        if self.filename and save_data:
            self.save_data(filename=self.filename, runtime=runtime)
        if postprocess:
            self.postprocess()
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
        print(f"Saved file to {filename}")
        return filename

    def __str__(self, *args, **kwargs):
        result_string = f"""
        {self.title} simulation ({os.path.basename(self.filename)}) containing {self.NT} iterations with time step {
        self.dt:.3e}
        Done on {self.run_date} from git version {self.git_version}
        {self.grid.NG}-cell grid of length {self.grid.L:.2f}. Epsilon zero = {self.grid.epsilon_0}, 
        c = {self.grid.c}""".lstrip()
        for species in self.list_species:
            result_string = result_string + "\n" + str(species)
        return result_string  # REFACTOR: add information from config file (run_coldplasma...)


def load_simulation(filename: str) -> Simulation:
    """
    Create a Simulation object from a hdf5 file.

    Parameters
    ----------
    filename : str
        Path to a hdf5 file.

    Returns
    -------
    Simulation
    """
    with h5py.File(filename, "r") as f:
        total_energy = f['Total energy'][...]
        title = f.attrs['title']
        grid_data = f['grid']
        grid = load_grid(grid_data, postprocess=True)

        all_species = [load_species(f['species'][species_group_name], grid, postprocess=True)
                       for species_group_name in f['species']]
        run_date = f.attrs['run_date']
        git_version = f.attrs['git_version']
    S = Simulation(grid, all_species, run_date=run_date, git_version=git_version, filename=filename, title=title)
    S.postprocessed = True

    S.total_energy = total_energy

    return S
