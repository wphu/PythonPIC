"""Data interface class"""
# coding=utf-8
import os
import time

import h5py
import numpy as np

from .grid import Grid, load_grid
from .species import load_species
from ..helper_functions.helpers import report_progress, git_version, config_filename


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
                 filename=time.strftime("%Y-%m-%d_%H-%M-%S"), category_type=None, config_version=None, title="",
                 considered_large=False):
        self.NT = grid.NT
        self.dt = grid.dt
        self.t = np.arange(self.NT) * self.dt
        self.grid = grid
        if list_species is None:
            list_species = []
        self.list_species = list_species
        self.field_energy = np.zeros(self.NT)
        self.filename = config_filename(filename, category_type, config_version)
        self.title = title
        self.git_version = git_version
        self.run_date = run_date

        self.postprocessed=False
        self.runtime = None
        self.considered_large = considered_large

    def postprocess(self):
        if not self.postprocessed:
            self.grid.postprocess()
            self.total_kinetic_energy = np.zeros(self.NT)
            for species in self.list_species:
                species.postprocess()
                self.total_kinetic_energy += species.kinetic_energy_history
            print("Postprocessing simulation.")
            self.total_energy =  self.total_kinetic_energy + self.grid.grid_energy_history
            self.postprocessed = True
        return self

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
            species.init_push(self.grid.field_function)
        return self

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
        self.grid.save_field_values(i)

        for species in self.list_species:
            species.save_particle_values(i)
            species.push(self.grid.field_function)
            species.apply_bc()
        self.grid.apply_bc(i)
        self.grid.gather_charge(self.list_species)
        self.grid.gather_current(self.list_species)
        self.grid.solve()

    def run(self, init=True) -> float:
        """
        Run n iterations of the simulation, saving data as it goes.

        Also measures runtime, saving it in self.runtime as a float with units of seconds.

        Parameters
        ----------
        init : bool
            Whether or not to initialize the simulation (particle placement, grid interaction).
            Not necessary, for example, in some tests.

        Returns
        -------
        self: Simulation
            The simulation, for chaining purposes.
        """
        if init:
            self.grid_species_initialization()
        start_time = time.time()
        for i in range(self.NT):
            if self.considered_large and i % (self.NT // 100) == 0:
                report_progress(i, self.NT, start_time)
            self.iteration(i)
        self.runtime = time.time() - start_time
        return self

    def lazy_run(self):
        """Does a simulation run() unless there's already a saved data with that file.

        If that file contains the same initial conditions and config version, the simulation's results are
        loaded instead.

        If the simulation errors during loading, it is ran anew."""
        print(f"Path is {self.filename}")
        file_exists = os.path.isfile(self.filename)
        if file_exists:
            print("Found file. Attempting to load...")
            try:
                loaded = load_simulation(self.filename)
                print("Managed to load file.")
                if loaded == self:
                    return loaded.postprocess()
                else:
                    print("Simulation files differ.")
            except KeyError as err:
                print(err)
        print("Running simulation")
        return self.run().save_data().postprocess()

    def test_run(self):
        """Does a blind run without saving data, for test purposes."""
        return self.run().postprocess()


    def save_data(self):
        """Save simulation data to hdf5.
        filename by default is the timestamp for the simulation."""
        if not os.path.exists(os.path.dirname(self.filename)):
            os.makedirs(os.path.dirname(self.filename))
        with h5py.File(self.filename, "w") as f:
            grid_data = f.create_group('grid')
            self.grid.save_to_h5py(grid_data)

            all_species = f.create_group('species')
            for species in self.list_species:
                species_data = all_species.create_group(species.name)
                species.save_to_h5py(species_data)

            f.attrs['dt'] = self.dt
            f.attrs['NT'] = self.NT
            f.attrs['run_date'] = self.run_date
            f.attrs['git_version'] = self.git_version
            f.attrs['title'] = self.title
            f.attrs['runtime'] = self.runtime
            f.attrs['considered_large'] = self.considered_large
        print(f"Saved file to {self.filename}")
        return self

    def __str__(self, *args, **kwargs):
        result_string = f"""
        {self.title} simulation ({os.path.basename(self.filename)}) containing {self.NT} iterations with time step {
        self.dt:.3e}
        Done on {self.run_date} from git version {self.git_version}
        {self.grid.NG}-cell grid of length {self.grid.L:.2f}. Epsilon zero = {self.grid.epsilon_0}, 
        c = {self.grid.c}""".strip()
        for species in self.list_species:
            result_string = result_string + "\n" + str(species)
        return result_string  # REFACTOR: add information from config file (run_coldplasma...)

    def __eq__(self, other):
        return True # TODO: compare

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
        title = f.attrs['title']
        grid_data = f['grid']
        grid = load_grid(grid_data, postprocess=True)

        all_species = [load_species(f['species'][species_group_name], grid, postprocess=True)
                       for species_group_name in f['species']]
        run_date = f.attrs['run_date']
        git_version = f.attrs['git_version']
        considered_large = f.attrs['considered_large']
    S = Simulation(grid, all_species, run_date=run_date, git_version=git_version, filename=filename, title=title, considered_large=considered_large)
    S.filename = filename

    S.postprocess()
    return S
