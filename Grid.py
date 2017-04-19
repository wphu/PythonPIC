"""The spatial grid"""
# coding=utf-8
import numpy as np
import scipy.fftpack as fft

import BoundaryCondition
import FieldSolver
import algorithms_interpolate


class Grid:
    """Object representing the grid on which charges and fields are computed
    """

    def __init__(self, L: float = 2 * np.pi, NG: int = 32, epsilon_0: float = 1, NT: float = 1, c: float = 1,
                 dt: float = 1, n_species: int = 1, solver=FieldSolver.FourierSolver, bc=BoundaryCondition.PeriodicBC):
        """
        :param float L: grid length, in nondimensional units
        :param int NG: number of grid cells
        :param float epsilon_0: the physical constant
        :param int NT: number of timesteps for history tracking purposes
        """
        self.x, self.dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
        self.dt = dt
        self.charge_density = np.zeros(NG + 2)
        self.current_density = np.zeros((NG + 2, 3))
        self.electric_field = np.zeros((NG + 2, 3))
        self.magnetic_field = np.zeros((NG + 2, 2))
        self.energy_per_mode = np.zeros(int(NG / 2))

        self.L = L
        self.NG = int(NG)
        self.NT = NT

        self.c = c
        self.epsilon_0 = epsilon_0
        self.n_species = n_species

        self.charge_density_history = np.zeros((NT, self.NG, n_species))
        self.current_density_history = np.zeros((NT, self.NG, 3, n_species))
        self.electric_field_history = np.zeros((NT, self.NG, 3))
        self.magnetic_field_history = np.zeros((NT, self.NG, 2))

        self.energy_per_mode_history = np.zeros(
            (NT, int(self.NG / 2)))  # OPTIMIZE: get this from efield_history?
        self.grid_energy_history = np.zeros(NT)  # OPTIMIZE: get this from efield_history

        self.solver_string = solver
        self.bc_string = bc

        # specific to Poisson solver but used also elsewhere, for plotting # TODO: clear this part up
        self.k = 2 * np.pi * fft.fftfreq(NG, self.dx)
        self.k[0] = 0.0001
        self.k_plot = self.k[:int(NG / 2)]

        self.solver = solver
        self.bc_function = bc.field_bc

    def init_solver(self):
        return self.solver.init_solver(self)

    def solve(self):
        return self.solver.solve(self)

    def direct_energy_calculation(self):
        r"""
        Direct energy calculation as

        :math:`E = \frac{\epsilon_0}{2} \sum_{i=0}^{NG} E^2 \Delta x`

        :return float E: calculated energy
        """
        return self.epsilon_0 * (self.electric_field ** 2).sum() * 0.5 * self.dx

    def apply_bc(self, i):
        bc_value = self.bc_function(i * self.dt)
        if bc_value:
            self.electric_field[0, 1] = bc_value

    def gather_charge(self, list_species, i=0):
        self.charge_density[:] = 0.0
        for i_species, species in enumerate(list_species):
            gathered_density = algorithms_interpolate.charge_density_deposition(self.x, self.dx, species.x[species.alive],
                                                                                species.q)
            self.charge_density_history[i, :, i_species] = gathered_density
            self.charge_density[1:-1] += gathered_density

    def gather_current(self, list_species, i=0):
        self.current_density[:] = 0.0
        for i_species, species in enumerate(list_species):
            gathered_current = algorithms_interpolate.current_density_deposition(self.x, self.dx, species.x, species.q,
                                                                                 species.v)
            self.current_density_history[i, :, :, i_species] = gathered_current
            self.current_density[1:-1] += gathered_current

    def electric_field_function(self, xp):
        # TODO: this only takes x right now
        return algorithms_interpolate.interpolateField(xp, self.electric_field[1:-1, 0], self.x,
                                                       self.dx)  # OPTIMIZE: this is probably slow

    def save_field_values(self, i):
        """Update the i-th set of field values, without those gathered from interpolation (charge\current)"""
        self.electric_field_history[i] = self.electric_field[1:-1]
        self.magnetic_field_history[i] = self.magnetic_field[1:-1]
        self.energy_per_mode_history[i] = self.energy_per_mode
        self.grid_energy_history[i] = self.energy_per_mode.sum() / (self.NG / 2)

    def save_to_h5py(self, grid_data):
        """
        Saves all grid data to h5py file
        grid_data: h5py group in premade hdf5 file
        """

        grid_data.attrs['NGrid'] = self.NG
        grid_data.attrs['L'] = self.L
        grid_data.attrs['epsilon_0'] = self.epsilon_0
        grid_data.create_dataset(name="x", dtype=float, data=self.x)

        grid_data.create_dataset(name="rho", dtype=float, data=self.charge_density_history)
        grid_data.create_dataset(name="current", dtype=float, data=self.current_density_history)
        grid_data.create_dataset(name="Efield", dtype=float, data=self.electric_field_history)
        grid_data.create_dataset(name="Bfield", dtype=float, data=self.magnetic_field_history)

        grid_data.create_dataset(name="energy per mode", dtype=float,
                                 data=self.energy_per_mode_history)  # OPTIMIZE: do these in post production
        grid_data.create_dataset(name="grid energy", dtype=float, data=self.grid_energy_history)

    def load_from_h5py(self, grid_data):
        """
        Loads all grid data from h5py file
        grid_data: h5py group in premade hdf5 file
        """
        self.NG = grid_data.attrs['NGrid']
        self.L = grid_data.attrs['L']
        self.epsilon_0 = grid_data.attrs['epsilon_0']
        self.NT = grid_data['rho'].shape[0]

        # OPTIMIZE: check whether these might not be able to be loaded partially for animation...?
        self.x = grid_data['x'][...]
        self.dx = self.x[1] - self.x[0]
        self.charge_density_history = grid_data['rho'][...]
        self.current_density_history = grid_data['current'][...]
        self.electric_field_history = grid_data['Efield'][...]
        self.magnetic_field_history = grid_data['Bfield'][...]
        self.energy_per_mode_history = grid_data["energy per mode"][
            ...]  # OPTIMIZE: this can be calculated during analysis
        self.grid_energy_history = grid_data["grid energy"][...]


