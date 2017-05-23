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
        self.scaling = scaling
        self.eff_q = q * scaling
        self.eff_m = m * scaling

        self.dt = grid.dt
        self.NT = grid.NT

        self.save_every_n_iterations = helper_functions.calculate_particle_iter_step(grid.NT)
        self.saved_iterations = helper_functions.calculate_particle_snapshots(grid.NT)
        self.x = np.zeros(N, dtype=float)
        self.v = np.zeros((N, 3), dtype=float)
        self.alive = np.ones(N, dtype=bool)
        self.c = grid.c
        self.name = name
        if self.N >= MAX_SAVED_PARTICLES:
            self.save_every_n_particle = (self.N // MAX_SAVED_PARTICLES)
            self.saved_particles = self.N // self.save_every_n_particle
            print(f"Too many macro{name} to save them all! N: {self.N}, so we're saving every "
                  f"{self.save_every_n_particle}th one and we're going to have "
                  f"{self.saved_particles}"
                  f" of them")
        else:
            self.saved_particles = self.N
            self.save_every_n_particle = 1

        self.position_history = np.zeros((self.saved_iterations, self.saved_particles))
        self.velocity_history = np.zeros((self.saved_iterations, self.saved_particles, 3))
        self.velocity_mean_history = np.zeros((self.saved_iterations, 3))
        self.velocity_std_history = np.zeros((self.saved_iterations, 3))
        self.alive_history = np.zeros((self.saved_iterations, self.saved_particles), dtype=bool)
        self.kinetic_energy_history = np.zeros(self.NT)
        self.pusher = pusher

    def init_push(self, electric_field_function, magnetic_field_function=lambda x: np.zeros((x.size, 3))):
        r"""
        Initializes particles for Leapfrog pushing.
        Same as `leapfrog_push`, except
        a) doesn't move particles in space,
        b) uses -dt/2

        :param electric_field_function: E(x), interpolated from grid
        :param float dt: original time step.
        :return float energy: (N,) size array of particle kinetic energies calculated at half time step
        """

        E = electric_field_function(self.x[self.alive])
        B = magnetic_field_function(self.x[self.alive])
        _, self.v[self.alive], energy = self.pusher(self, E, -self.dt * 0.5, B)
        return energy

    def push(self, electric_field_function, magnetic_field_function=lambda x: np.zeros((x.size, 3))):
        r"""
        Leapfrog pusher for particles.

        :param electric_field_function: E(x), interpolated from grid
        :param float dt: original time step
        :return float energy: (N,) size array of particle kinetic energies calculated at half time step
        """

        E = electric_field_function(self.x[self.alive])
        B = magnetic_field_function(self.x[self.alive])
        self.x[self.alive], self.v[self.alive], energy = self.pusher(self, E, self.dt, B)
        return energy

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
        self.x %= L  # ensure boundaries # TODO: this will not work with non-periodic boundary conditions

    def random_position_perturbation(self, std: float):
        self.x += np.random.normal(scale=std, size=self.N)

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
            index = helper_functions.convert_global_to_particle_iter(i, self.save_every_n_iterations)
            self.position_history[index] = self.x[::self.save_every_n_particle]
            self.velocity_history[index] = self.v[::self.save_every_n_particle]
            self.velocity_mean_history[index] = self.v[self.alive].mean(axis=0)
            self.velocity_std_history[index] = self.v[self.alive].std(axis=0)

    def save_to_h5py(self, species_data):
        """
        Saves all species data to h5py file
        species_data: h5py group for this species in pre-made hdf5 file
        """
        species_data.attrs['name'] = self.name
        species_data.attrs['N'] = self.N
        species_data.attrs['q'] = self.q
        species_data.attrs['m'] = self.m

        species_data.create_dataset(name="x", dtype=float, data=self.position_history)
        species_data.create_dataset(name="v", dtype=float, data=self.velocity_history)
        species_data.create_dataset(name="alive", dtype=float, data=self.alive)
        species_data.create_dataset(name="Kinetic energy", dtype=float, data=self.kinetic_energy_history)

        species_data.create_dataset(name="v_mean", dtype=float, data=self.velocity_mean_history)
        species_data.create_dataset(name="v_std", dtype=float, data=self.velocity_std_history)

    def load_from_h5py(self, species_data):
        # REFACTOR: move this out of class (like Simulation.load_data)
        """
        Loads species data from h5py file
        species_data: h5py group for this species in pre-made hdf5 file
        """
        self.name = species_data.attrs['name']
        self.N = species_data.attrs['N']
        self.q = species_data.attrs['q']
        self.m = species_data.attrs['m']

        self.velocity_mean_history = species_data["v_mean"][...]
        self.velocity_std_history = species_data["v_std"][...]

        self.position_history = species_data["x"][...]
        self.velocity_history = species_data["v"][...]
        self.alive_history = species_data["alive"][...]
        self.kinetic_energy_history = species_data["Kinetic energy"][...]

    def __repr__(self, *args, **kwargs):
        return f"Species(q={self.q:.4f},m={self.m:.4f},N={self.N},name=\"{self.name}\",NT={self.NT})"

    def __str__(self):
        return f"{self.N} {self.scaling}-{self.name} with q = {self.q:.4f}, m = {self.m:.4f}, {self.saved_iterations} saved history " \
               f"steps over {self.NT} iterations"


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


if __name__ == '__main__':
    p = Particle(1, 3, 4, -5, name="test particle")
    print(p)
