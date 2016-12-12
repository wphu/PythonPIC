import numpy as np
import h5py
from FourierSolver import PoissonSolver
from scatter import charge_density_deposition
from gather import interpolateField
import scipy.fftpack as fft


class Grid(object):
    def __init__(self, L=2 * np.pi, NG=32, epsilon_0=1, NT=None):
        self.x, self.dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
        self.charge_density = np.zeros_like(self.x)
        self.electric_field = np.zeros_like(self.x)
        self.potential = np.zeros_like(self.x)
        self.energy_per_mode = np.zeros(int(NG / 2))
        self.L = L
        self.NG = int(NG)
        self.NT = NT
        self.epsilon_0 = epsilon_0
        self.k = 2 * np.pi * fft.fftfreq(NG, self.dx)
        self.k[0] = 0.0001
        self.k_plot = self.k[:int(NG / 2)]

        if NT:
            self.charge_density_history = np.zeros((NT, self.NG))
            self.electric_field_history = np.zeros((NT, self.NG))
            self.potential_history = np.zeros((NT, self.NG))
            self.mode_energy_history = np.zeros((NT, self.NG))
            self.energy_per_mode_history = np.zeros((NT, int(self.NG / 2)))
            self.grid_energy_history = np.zeros(NT)

    def solve_poisson(self):
        self.electric_field, self.potential, self.energy_per_mode = PoissonSolver(self.charge_density, self.k, self.NG, epsilon_0=self.epsilon_0)
        return self.energy_per_mode.sum() / (self.NG/2)# * 8 * np.pi * self.k[1]**2

    def gather_charge(self, list_species):
        self.charge_density[:] = 0.0
        for species in list_species:
            self.charge_density += charge_density_deposition(self.x, self.dx, species.x, species.q)

    def electric_field_function(self, xp):
        return interpolateField(xp, self.electric_field, self.x, self.dx)

    def save_field_values(self, i):
        """Update the i-th set of field values"""
        self.charge_density_history[i] = self.charge_density
        self.electric_field_history[i] = self.electric_field
        self.potential_history[i] = self.electric_field
        self.energy_per_mode_history[i] = self.energy_per_mode
        self.grid_energy_history[i] = self.energy_per_mode.sum() / (self.NG/2)

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
        grid_data.create_dataset(name="Efield", dtype=float, data=self.electric_field_history)
        grid_data.create_dataset(name="potential", dtype=float, data=self.potential_history)
        grid_data.create_dataset(name="energy per mode", dtype=float, data=self.energy_per_mode_history)
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

        # TODO: check whether these might not be able to be loaded partially for animation...?
        self.x = grid_data['x'][...]
        self.dx = self.x[1] - self.x[0]
        self.charge_density_history = grid_data['rho'][...]
        self.electric_field_history = grid_data['Efield'][...]
        self.potential_history = grid_data["potential"][...]
        self.energy_per_mode_history = grid_data["energy per mode"][...]
        self.grid_energy_history = grid_data["grid energy"][...]

    def __eq__(self, other):
        result = True
        result *= np.isclose(self.x, other.x).all()
        result *= np.isclose(self.charge_density, other.charge_density).all()
        result *= np.isclose(self.electric_field, other.electric_field).all()
        result *= np.isclose(self.potential, other.potential).all()
        result *= self.dx == other.dx
        result *= self.L == other.L
        result *= self.NG == other.NG
        result *= self.epsilon_0 == other.epsilon_0
        return result
    # def plot(self, show=True):
    #     plt.plot(self.x, self.charge_density
