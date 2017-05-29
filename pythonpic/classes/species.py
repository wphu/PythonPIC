"""Class representing a group of particles"""
# coding=utf-8
import numpy as np

from ..algorithms import density_profiles, helper_functions
from ..algorithms.particle_push import rela_boris_push_bl as rela_boris_push

MAX_SAVED_PARTICLES = int(1e4)


class Species:
    """
    Object representing a species of particles: ions, electrons, or simply
    a group of particles with a particular initial velocity distribution.

    Parameters
    ----------
    q : float
        particle charge
    m : float
        particle mass
    N : int
        number of macroparticles
    grid : Grid or Frame
        parent grid
    name : str
        name of group
    scaling : float
        number of particles per macroparticle
    pusher : function 
        particle push algorithm
    """
    def __init__(self, q, m, N, grid, name="particles", scaling=1, pusher=rela_boris_push):
        self.q = q
        self.m = m
        self.N = int(N)
        self.N_alive = N
        self.scaling = scaling
        self.eff_q = q * scaling
        self.eff_m = m * scaling

        self.grid = grid
        self.particle_bc = grid.particle_bc
        self.dt = grid.dt
        self.NT = grid.NT
        self.c = grid.c

        self.save_every_n_iterations = helper_functions.calculate_particle_iter_step(grid.NT)
        self.saved_iterations = helper_functions.calculate_particle_snapshots(grid.NT)
        self.x = np.zeros(N, dtype=float)
        self.v = np.zeros((N, 3), dtype=float)
        self.energy = self.kinetic_energy()
        self.alive = np.ones(N, dtype=bool)
        self.name = name
        if self.N >= MAX_SAVED_PARTICLES:
            self.save_every_n_particle = (self.N // MAX_SAVED_PARTICLES)
            self.saved_particles = np.ceil(self.N / self.save_every_n_particle).astype(int)
            print(f"Too many macro{name} to save them all! N: {self.N}, so we're saving every "
                  f"{self.save_every_n_particle}th one and we're going to have "
                  f"{self.saved_particles}"
                  f" of them")
        else:
            self.saved_particles = self.N
            self.save_every_n_particle = 1

        self.position_history = np.zeros((self.saved_iterations, self.saved_particles), dtype=float)
        self.velocity_history = np.zeros((self.saved_iterations, self.saved_particles, 3), dtype=float)
        self.velocity_mean_history = np.zeros((self.saved_iterations, 3), dtype=float)
        self.velocity_std_history = np.zeros((self.saved_iterations, 3), dtype=float)
        # self.alive_history = np.zeros((self.saved_iterations, self.saved_particles), dtype=bool)
        self.N_alive_history = np.zeros(self.saved_iterations, dtype=int)
        self.number_alive_history = np.zeros((self.saved_iterations), dtype=int)
        self.kinetic_energy_history = np.zeros(self.NT+1)
        self.pusher = pusher

    def apply_bc(self):
        self.particle_bc(self)

    def kinetic_energy(self):
        return 0.5 * self.m * np.sum(self.v**2) # TODO: make this relativistic

    def init_push(self, electric_field_function, magnetic_field_function=lambda x: np.zeros((x.size, 3))):
        """
        Push the particles using the previously set pushing algorithm.
        This is the same thing as seen in `push`, except that it doesn't update positions.
        That is necessary for energy conservation purposes of Boris and Leapfrog pushers.

        Parameters
        ----------
        electric_field_function : ndarray
        magnetic_field_function : ndarray
            Arrays of interpolated field values. Shape should be (N_particles, 3).

        Returns
        -------

        The kinetic energy of the particles, calculated at half timestep.
        """

        E = electric_field_function(self.x)
        B = magnetic_field_function(self.x)
        _, self.v, self.energy = self.pusher(self, E, -self.dt * 0.5, B)
        return self.energy

    def push(self, electric_field_function, magnetic_field_function=lambda x: np.zeros((x.size, 3))):
        """
        Push the particles using the previously set pushing algorithm.

        Parameters
        ----------
        electric_field_function : ndarray
        magnetic_field_function : ndarray
            Arrays of interpolated field values. Shape should be (N_particles, 3).

        Returns
        -------

        The kinetic energy of the particles, calculated at half timestep.
        """
        E = electric_field_function(self.x)
        B = magnetic_field_function(self.x)
        self.x, self.v, self.energy = self.pusher(self, E, self.dt, B)
        return self.energy

    """POSITION INITIALIZATION"""

    def distribute_uniformly(self, Lx: float, shift: float = 0, start_moat=0, end_moat=0):
        """
        Distribute uniformly on grid.

        :param Lx: grid size
        :param shift: displace all particles right by this distance
        """
        self.x = (np.linspace(start_moat + Lx / self.N * 1e-10, Lx - end_moat, self.N,
                              endpoint=False) + shift * self.N / Lx / 10) % Lx  # Type:

    def distribute_nonuniformly(self, L, moat_length, ramp_length, plasma_length, resolution_increase=1000,
                                profile="linear"):
        dense_x = np.linspace(moat_length*0.95, (moat_length + plasma_length)*1.05, self.N * resolution_increase)
        self.x = density_profiles.generate(dense_x, density_profiles.FDENS, moat_length,
                                           ramp_length,
                                           plasma_length, self.N, profile)

    def sinusoidal_position_perturbation(self, amplitude: float, mode: int, L: float):
        """
        Displace positions by a sinusoidal perturbation calculated for each particle.
        
            dx = amplitude * cos(2 * mode * pi * x / L)xk

        :param float amplitude: Amplitude of perturbation
        :param int mode: which mode is excited/home/dominik/Inzynierka/pythonpic/pythonpic/tests/__init__.py
        :param float L: grid length
        :return:
        """
        self.x += amplitude * np.cos(2 * mode * np.pi * self.x / L)
        self.apply_bc()

    def random_position_perturbation(self, std: float):
        self.x += np.random.normal(scale=std, size=self.N)
        self.apply_bc()

    """VELOCITY INITIALIZATION"""

    def sinusoidal_velocity_perturbation(self, axis: int, amplitude: float, mode: int, L: float):
        """
        Displace velocities by a sinusoidal perturbation calculated for each particle.
        

        :param int axis: axis, for 3d velocities
        :param float amplitude: of perturbation
        :param int mode: which mode is excited
        :param float L: grid length
        """
        self.v[:, axis] += amplitude * np.cos(2 * mode * np.pi * self.x / L)

    def random_velocity_perturbation(self, axis: int, std: float):
        """
        Add Gausian noise to particle velocities on

        :param int axis:
        :param float std: standard deviation of noise
        """
        self.v[:, axis] += np.random.normal(scale=std, size=self.N)

    # def init_velocity_maxwellian(self, T, resolution_increase = 1000):
    #     thermal_velocity = 1
    #     dense_p = np.linspace(0, 4 * thermal_velocity, self.N/4 * 1000)
    #
    #     # TODO: WORK IN PROGRESS
    #     self.v = result

    """ DATA ACCESS """

    def save_particle_values(self, i: int):
        """Update the i-th set of particle values"""
        if helper_functions.is_this_saved_iteration(i, self.save_every_n_iterations):
            N_alive = self.x.size
            save_every_n_particle = (N_alive // MAX_SAVED_PARTICLES)
            save_every_n_particle = 1 if save_every_n_particle == 0 else save_every_n_particle
            index = helper_functions.convert_global_to_particle_iter(i, self.save_every_n_iterations)
            self.position_history[index, :N_alive] = self.x[::save_every_n_particle]
            self.velocity_history[index, :N_alive] = self.v[::save_every_n_particle]
            self.velocity_mean_history[index] = self.v.mean(axis=0)
            self.velocity_std_history[index] = self.v.std(axis=0)
            self.number_alive_history[index] = N_alive
        self.kinetic_energy_history[i] = self.energy

    def save_to_h5py(self, species_data):
        """
        Saves all species data to h5py file

        Parameters
        ----------
        species_data : h5py group
            h5py group for this species in pre-made hdf5 file
        Returns
        -------

        """
        species_data.attrs['name'] = self.name
        species_data.attrs['N'] = self.N
        species_data.attrs['q'] = self.q
        species_data.attrs['m'] = self.m
        species_data.attrs['scaling'] = self.scaling

        species_data.create_dataset(name="x", dtype=float, data=self.position_history)
        species_data.create_dataset(name="v", dtype=float, data=self.velocity_history)
        species_data.create_dataset(name="Kinetic energy", dtype=float, data=self.kinetic_energy_history)

        species_data.create_dataset(name="v_mean", dtype=float, data=self.velocity_mean_history)
        species_data.create_dataset(name="v_std", dtype=float, data=self.velocity_std_history)
        species_data.create_dataset(name="N_alive_history", dtype=int, data=self.N_alive_history)

    def postprocess(self):
        pass # TODO: implement

    def __repr__(self, *args, **kwargs):
        return f"Species(q={self.q:.4f},m={self.m:.4f},N={self.N},name=\"{self.name}\",NT={self.NT})"

    def __str__(self):
        return f"{self.N} {self.scaling:.2e}-{self.name} with q = {self.q:.2e}, m = {self.m:.2e}, {self.saved_iterations} saved history " \
               f"steps over {self.NT} iterations"

def load_species(species_data, grid, postprocess=False):
    """
    Loads species data from h5py file.
    Parameters
    ----------
    species_data : h5py path
        Path in open hdf5 file
    grid : Grid
        grid to load particles onto
    postprocess : bool
        Whether to run additional processing
    Returns
    -------

    """
    name = species_data.attrs['name']
    N = species_data.attrs['N']
    q = species_data.attrs['q']
    m = species_data.attrs['m']
    scaling = species_data.attrs['scaling']


    species = Species(q, m, N, grid, name, scaling)
    species.velocity_mean_history = species_data["v_mean"][...]
    species.velocity_std_history = species_data["v_std"][...]

    species.position_history = species_data["x"][...]
    species.velocity_history = species_data["v"][...]
    species.N_alive_history = species_data["N_alive_history"][...]
    species.kinetic_energy_history = species_data["Kinetic energy"][...]
    if postprocess:
        species.postprocess()
    return species


class Particle(Species):
    """
    A helper class for quick creation of a single particle for test purposes.
    Parameters
    ----------
    grid : Grid or Frame
        parent grid
    x : float
        position
    vx : float
        x velocity
    vy : float
        y velocity
    vz : float 
        z velocity
    q : float
        particle charge
    m : float
        particle mass
    name : str
        name of group
    scaling : float
        number of particles per macroparticle
    pusher : function 
        particle push algorithm
    """
    def __init__(self, grid, x, vx, vy=0, vz=0, q=1, m=1, name="Test particle", scaling=1, pusher=rela_boris_push):
        # noinspection PyArgumentEqualDefault
        super().__init__(q, m, 1, grid, name, scaling = scaling, pusher=pusher)
        self.x[:] = x
        self.v[:, 0] = vx
        self.v[:, 1] = vy
        self.v[:, 2] = vz

