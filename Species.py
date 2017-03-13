"""Class representing a group of particles"""
# coding=utf-8
import numpy as np

from algorithms_pusher import rela_boris_push

MAX_SAVED_PARTICLES = int(1e4)


class Species:
    """Object representing a species of particles: ions, electrons, or simply
    a group of particles with a particular (nyeh) initial velocity distribution.

    q: float, particle charge
    m: float, particle mass
    N: int, total number of particles in species
    name: string, ID of particles
    NT: int, number of time steps (for diagnostics)
    """

    def __init__(self, q, m, N, name=None, NT=1):
        r"""
        :param float q: particle charge
        :param float m: particle mass
        :param int N: total number of species particles
        :param str name: name of particle set
        :param int NT: number of time steps (for history saving)
        """
        self.q = q
        self.m = m
        self.N = int(N)
        self.NT = NT
        self.x = np.zeros(N, dtype=float)
        self.v = np.zeros((N, 3), dtype=float)
        if name:
            self.name = name
        else:
            self.name = "q{}m{}N{}".format(q, m, N)
        if self.N >= MAX_SAVED_PARTICLES:
            self.saved_particles = MAX_SAVED_PARTICLES
            self.save_every_n_particle = self.N // MAX_SAVED_PARTICLES
        else:
            self.saved_particles = self.N
            self.save_every_n_particle = 1

        self.position_history = np.zeros((NT, self.saved_particles))
        self.velocity_history = np.zeros((NT, self.saved_particles, 3))
        self.kinetic_energy_history = np.zeros(NT)

    def init_push(self, electric_field_function, dt, *args):
        r"""
        Initializes particles for Leapfrog pushing.
        Same as `leapfrog_push`, except
        a) doesn't move particles in space,
        b) uses -dt/2

        :param electric_field_function: E(x), interpolated from grid
        :param float dt: original time step.
        :return float energy: (N,) size array of particle kinetic energies calculated at half time step
        """

        electric_force = electric_field_function(self.x) * self.q / self.m
        v_new = self.v.copy()
        v_new[:, 0] -= electric_force * 0.5 * dt
        energy = self.v * v_new * (0.5 * self.m)
        self.v = v_new
        return energy

    def push(self, electric_field_function, dt, *args):
        r"""
        Leapfrog pusher for particles.

        :param electric_field_function: E(x), interpolated from grid
        :param float dt: original time step
        :param float L: grid size, used to enforce boundary condition
        :return float energy: (N,) size array of particle kinetic energies calculated at half time step
        """
        electric_force = electric_field_function(self.x) * self.q / self.m
        v_new = self.v.copy()
        v_new[:, 0] += electric_force * dt

        self.x += v_new[:, 0] * dt
        energy = self.v * v_new * (0.5 * self.m)
        self.v = v_new
        return energy

    def return_to_bounds(self, L):
        self.x %= L

    """POSITION INITIALIZATION"""

    def distribute_uniformly(self, Lx: float, shift: float = 0):
        """
        Distribute uniformly on grid.

        :param Lx: grid size
        :param shift: displace all particles right by this distance
        """
        self.x = (np.linspace(Lx / self.N / 1e10, Lx, self.N, endpoint=False) + shift * self.N / Lx / 10) % Lx  # Type:

    def sinusoidal_position_perturbation(self, amplitude: float, mode: int, L: float):
        """
        Displace positions by a sinusoidal perturbation calculated for each particle.

        :param float amplitude: Amplitude of perturbation
        :param int mode: which mode is excited
        :param float L: grid length
        :return:
        """
        self.x += amplitude * np.cos(2 * mode * np.pi * self.x / L)
        self.x %= L  # ensure boundaries

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

    """ DATA ACCESS """

    def save_particle_values(self, i: int):
        """Update the i-th set of particle values"""
        self.position_history[i] = self.x[::self.save_every_n_particle]
        self.velocity_history[i] = self.v[::self.save_every_n_particle]

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
        species_data.create_dataset(name="Kinetic energy", dtype=float, data=self.kinetic_energy_history)

    def load_from_h5py(self, species_data):
        # TODO: move this out of class
        """
        Loads species data from h5py file
        species_data: h5py group for this species in pre-made hdf5 file
        """
        self.name = species_data.attrs['name']
        self.N = species_data.attrs['N']
        self.q = species_data.attrs['q']
        self.m = species_data.attrs['m']

        self.position_history = species_data["x"][...]
        self.velocity_history = species_data["v"][...]
        self.kinetic_energy_history = species_data["Kinetic energy"][...]

    def __repr__(self, *args, **kwargs):
        return f"Species(q={self.q:.4f},m={self.m:.4f},N={self.N},name=\"{self.name}\",NT={self.NT})"

    def __eq__(self, other):
        result = True
        result *= self.q == other.q
        result *= self.m == other.m
        result *= self.N == other.N
        result *= np.isclose(self.x, other.x).all()
        result *= np.isclose(self.v, other.v).all()
        result *= self.name == other.name
        return result


class MagneticSpecies(Species):
    """Particle class for magnetic simulations"""

    def init_push(self, electric_field_function, dt, magnetic_field_function, *args):
        """Boris pusher initialization, nonrelativistic"""

        dt = -dt / 2
        # add half electric impulse to v(t-dt/2)
        efield = np.zeros((self.N, 3))
        efield[:, 0] = electric_field_function(self.x)
        vminus = self.v + self.q * efield / self.m * dt * 0.5

        # rotate to add magnetic field
        t = -magnetic_field_function(self.x) * self.q / self.m * dt * 0.5
        s = 2 * t / (1 + t * t)

        vprime = vminus + np.cross(vminus, t)  # type: np.ndarray # TODO: axis?
        vplus = vminus + np.cross(vprime, s)  # type: np.ndarray
        v_new = vplus + self.q * efield / self.m * dt * 0.5  # type: np.ndarray

        energy = self.v * v_new * (0.5 * self.m)
        self.v = v_new
        return energy

    def push(self, electric_field_function, dt, magnetic_field_function, *args):
        """Boris pusher, nonrelativistic"""
        # add half electric impulse to v(t-dt/2)
        efield = np.zeros((self.N, 3))
        efield[:, 0] = electric_field_function(self.x)
        vminus = self.v + self.q * efield / self.m * dt * 0.5

        # rotate to add magnetic field
        t = -magnetic_field_function(self.x) * self.q / self.m * dt * 0.5
        s = 2 * t / (1 + t * t)

        vprime = vminus + np.cross(vminus, t)  # TODO: axis?
        vplus = vminus + np.cross(vprime, s)
        v_new = vplus + self.q * efield / self.m * dt * 0.5

        self.x += v_new[:, 0] * dt

        energy = self.v * v_new * (0.5 * self.m)
        self.v = v_new
        return energy


class RelativisticSpecies(Species):
    """Particle class for relativistic electromagnetic simulations"""

    def init_push(self, electric_field_function, dt, magnetic_field_function, c, *args):
        """Boris pusher initialization, relativistic"""

        E = electric_field_function(self.x)
        B = magnetic_field_function(self.x)
        _, self.v, energy = rela_boris_push(self.x, self.v, E, B, self.q, self.m, -dt / 2, c)
        return energy

    def push(self, electric_field_function, dt, magnetic_field_function, c, *args):
        """Boris pusher, relativistic"""
        E = electric_field_function(self.x)
        B = magnetic_field_function(self.x)
        self.x, self.v, energy = rela_boris_push(self.x, self.v, E, B, self.q, self.m, dt, c)
        return energy
