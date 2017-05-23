"""The spatial grid"""
# coding=utf-8
import numpy as np
import scipy.fftpack as fft

from ..algorithms import field_interpolation, helper_functions, FieldSolver, BoundaryCondition
from ..algorithms.field_interpolation import longitudinal_current_deposition, transversal_current_deposition

class Frame:
    """
    Parameters
    ----------
    dt : float 
        timestep
    c : float
        speed of light
    epsilon_0 : float 
        electric permittivity of vacuum
    NT : int
        number of timesteps, default 1
    """
    def __init__(self, dt: float, c: float, epsilon_0: float, NT: int = 1):
        self.dt = dt
        self.c = c
        self.epsilon_0 = epsilon_0
        self.NT = NT


class TimelessGrid(Frame):
    """
    A mock grid for tests with only constants. Does not depend on time.
    
    Parameters
    ----------
    L : float
        total length of simulation area
    NG : int
        number of grid cells
    c : float
        speed of light
    epsilon_0 : float 
        electric permittivity of vacuum
    bc : BoundaryCondition
    solver : FieldSolver
    NT : int
        number of timesteps for particle simulations, default 1
    """
    def __init__(self, L: float, NG: int, c: float = 1, epsilon_0: float = 1, bc=BoundaryCondition.PeriodicBC,
                 solver=FieldSolver.FourierSolver, NT = 1):

        self.x, self.dx = np.linspace(0, L, NG, retstep=True, endpoint=False)

        dt = self.dx / c
        super().__init__(dt, epsilon_0, c, NT)
        self.epsilon_0 = epsilon_0

        self.charge_density = np.zeros(NG + 2)
        self.current_density = np.zeros((NG + 2, 3))
        self.electric_field = np.zeros((NG + 2, 3))
        self.magnetic_field = np.zeros((NG + 2, 3))
        self.energy_per_mode = np.zeros(int(NG / 2))

        self.L = L
        self.NG = NG

        self.solver = solver
        self.bc_function = bc.field_bc

        # specific to Poisson solver but used also elsewhere, for plots # TODO: clear this part up
        self.k = 2 * np.pi * fft.fftfreq(self.NG, self.dx)
        self.k[0] = 0.0001
        self.k_plot = self.k[:int(self.NG / 2)]

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

    def gather_charge(self, list_species):
        self.charge_density[...] = 0.0
        for species in list_species:
            gathered_density = field_interpolation.charge_density_deposition(self.x, self.dx,
                                                                             species.x[species.alive],
                                                                             species.eff_q)
            self.charge_density[1:-1] += gathered_density

    def gather_current(self, list_species):
        self.current_density[...] = 0.0
        for species in list_species:
            time_array = np.ones(species.N) * self.dt
            longitudinal_current_deposition(self.current_density[:, 0], species.v[:, 0], species.x, self.dx, self.dt,
                                            species.eff_q)
            transversal_current_deposition(self.current_density[:, 1:], species.v, species.x, self.dx, self.dt,
                                           species.eff_q)

    def electric_field_function(self, xp):
        result = np.zeros((xp.size, 3))
        for i in range(3):
            result[:, i] = field_interpolation.interpolateField(xp, self.electric_field[1:-1, i], self.x, self.dx)
        return result

    def magnetic_field_function(self, xp):
        result = np.zeros((xp.size, 3))
        for i in range(1, 3):
            result[:, i] = field_interpolation.interpolateField(xp, self.magnetic_field[1:-1, i], self.x, self.dx)
        return result

class Grid(TimelessGrid):
    """
    Object representing the grid on which charges and fields are computed
    """


    def __init__(self, T: float, L: float, NG: int, c: float = 1, epsilon_0: float = 1, bc=BoundaryCondition.PeriodicBC,
                 solver=FieldSolver.FourierSolver):
        """
        
        Parameters
        ----------
        T : float
            total runtime of the simulation
        L : float
            total length of simulation area
        NG : int
            number of grid cells
        c : float
            speed of light
        epsilon_0 : float
            electric permittivity of vacuum
        bc : BoundaryCondition
        solver : FieldSolver
        """

        """
        :param float L: grid length, in nondimensional units
        :param int NG: number of grid cells
        :param float epsilon_0: the physical constant
        :param int NT: number of timesteps for history tracking purposes
        """

        super().__init__(L, NG, c, epsilon_0, bc, solver)
        self.T = T
        self.NT = helper_functions.calculate_number_timesteps(T, self.dt)


        self.charge_density_history = np.zeros((self.NT, self.NG))
        self.current_density_history = np.zeros((self.NT, self.NG, 3))
        self.electric_field_history = np.zeros((self.NT, self.NG, 3))
        self.magnetic_field_history = np.zeros((self.NT, self.NG, 2))

        self.energy_per_mode_history = np.zeros(
            (self.NT, int(self.NG / 2)))  # OPTIMIZE: get this from efield_history?
        self.grid_energy_history = np.zeros(self.NT)  # OPTIMIZE: get this from efield_history




    def apply_bc(self, i):
        bc_value = self.bc_function(i * self.dt)
        if bc_value:
            self.electric_field[0, 1] = bc_value


    def save_field_values(self, i):
        """Update the i-th set of field values, without those gathered from interpolation (charge\current)"""
        self.charge_density_history[i, :] = self.charge_density[1:-1]
        self.current_density_history[i, :, :] = self.current_density[1:-1]
        self.electric_field_history[i] = self.electric_field[1:-1]
        self.magnetic_field_history[i] = self.magnetic_field[1:-1, 1:]
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

class PostprocessedGrid(Grid):
    """
    Object representing the grid, with post-simulation computation and visualization capabilities.
    """
    def __init__(self, grid_data):
        """
        Builds a grid and runs computation on it.
        Parameters
        ----------
        grid_data: path to grid_data in open h5py file
        """
        self.NG = grid_data.attrs['NGrid']
        self.L = grid_data.attrs['L']
        self.epsilon_0 = grid_data.attrs['epsilon_0']
        self.NT = grid_data['rho'].shape[0]
        # TODO: call super()

        # OPTIMIZE: check whether these might not be able to be loaded partially for animation...?
        self.x = grid_data['x'][...]
        self.dx = self.x[1] - self.x[0]
        self.x_current = self.x + self.dx / 2
        self.charge_density_history = grid_data['rho'][...]
        self.current_density_history = grid_data['current'][...]
        self.electric_field_history = grid_data['Efield'][...]
        self.magnetic_field_history = grid_data['Bfield'][...]
        # OPTIMIZE: this can be calculated during analysis
        self.energy_per_mode_history = grid_data["energy per mode"][...]
        self.grid_energy_history = grid_data["grid energy"][...]
