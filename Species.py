import numpy as np

MAX_SAVED_PARTICLES = int(1e4)
class Species(object):
    def __init__(self, q, m, N, name=None, NT=None):
        """Object representing a species of particles: ions, electrons, or simply
        a group of particles with a particular (heh) initial velocity distribution.

        q: float, particle charge
        m: float, particle mass
        N: int, total number of particles in species
        name: string, ID of particles
        NT: int, number of timesteps (for diagnostics)
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

        if NT:
            self.position_history = np.zeros((NT, self.saved_particles))
            self.velocity_history = np.zeros((NT, self.saved_particles, 3))
            self.kinetic_energy_history = np.zeros(NT)


    def leapfrog_init(self, electric_field_function, dt):
        """Leapfrog pusher initialization
        dt: usual timestep, minus halving is done automatically"""

        electric_force = electric_field_function(self.x) * self.q / self.m
        v_new = self.v.copy()
        v_new[:,0] -= electric_force * 0.5 * dt
        energy = self.v * v_new * (0.5 * self.m)
        self.v = v_new
        return energy

    def leapfrog_push(self, electric_field_function, dt, L):
        """Leapfrog pusher"""
        electric_force = electric_field_function(self.x) * self.q / self.m
        v_new = self.v.copy()
        v_new[:,0] += electric_force * dt

        self.x += v_new[:,0] * dt
        self.x %= L
        energy = self.v * v_new * (0.5 * self.m)
        self.v = v_new
        return energy

    def boris_init(self, electric_field_function, magnetic_field_function,  dt, L):
        """Boris pusher initialization, unrelativistic"""

        dt = -dt/2
        # add half electric impulse to v(t-dt/2)
        efield = np.zeros((self.N, 3))
        efield[:,0] = electric_field_function(self.x)
        vminus = self.v + self.q * efield / self.m * dt * 0.5

        # rotate to add magnetic field
        t = -magnetic_field_function(self.x) * self.q / self.m * dt * 0.5
        s = 2*t/(1+t*t)

        vprime = vminus + np.cross(vminus, t) # TODO: axis?
        vplus = vminus + np.cross(vprime, s)
        v_new = vplus + self.q * efield / self.m * dt * 0.5

        energy = self.v * v_new * (0.5 * self.m)
        self.v = v_new
        return energy

    def boris_push_particles(self, electric_field_function, magnetic_field_function, dt, L):
        """Boris pusher, unrelativistic"""
        # add half electric impulse to v(t-dt/2)
        efield = np.zeros((self.N, 3))
        efield[:,0] = electric_field_function(self.x)
        vminus = self.v + self.q * efield / self.m * dt * 0.5

        # rotate to add magnetic field
        t = -magnetic_field_function(self.x) * self.q / self.m * dt * 0.5
        s = 2*t/(1+t*t)

        vprime = vminus + np.cross(vminus, t) # TODO: axis?
        vplus = vminus + np.cross(vprime, s)
        v_new = vplus + self.q * efield / self.m * dt * 0.5

        self.x += v_new[:,0] * dt

        self.x %= L
        energy = self.v * v_new * (0.5 * self.m)
        self.v = v_new
        return energy
        # add remaining half of electric impulse


    def distribute_uniformly(self, Lx, shift=False):
        self.x = (np.linspace(0, Lx, self.N, endpoint=False) + shift * self.N / Lx / 10) % Lx

    def sinusoidal_position_perturbation(self, amplitude, mode, L):
        self.x += amplitude * np.cos(2 * mode * np.pi * self.x / L)
        self.x %= L

    def sinusoidal_velocity_perturbation(self, axis, amplitude, mode, L):
        self.v[:,axis] += amplitude * np.cos(2 * mode * np.pi * self.x / L)

    def save_particle_values(self, i):
        """Update the i-th set of particle values"""
        self.position_history[i] = self.x[::self.save_every_n_particle]
        self.velocity_history[i] = self.v[::self.save_every_n_particle]

    def save_to_h5py(self, species_data):
        """
        Saves all species data to h5py file
        species_data: h5py group for this species in premade hdf5 file
        """
        species_data.attrs['name'] = self.name
        species_data.attrs['N'] = self.N
        species_data.attrs['q'] = self.q
        species_data.attrs['m'] = self.m

        species_data.create_dataset(name="x", dtype=float, data=self.position_history)
        species_data.create_dataset(name="v", dtype=float, data=self.velocity_history)
        species_data.create_dataset(name="Kinetic energy", dtype=float, data=self.kinetic_energy_history)

    def load_from_h5py(self, species_data):
        """
        Loads species data from h5py file
        species_data: h5py group for this species in premade hdf5 file
        """
        self.name = species_data.attrs['name']
        self.N = species_data.attrs['N']
        self.q = species_data.attrs['q']
        self.m = species_data.attrs['m']

        self.position_history = species_data["x"][...]
        self.velocity_history = species_data["v"][...]
        self.kinetic_energy_history = species_data["Kinetic energy"][...]

    def __eq__(self, other):
        result = True
        result *= self.q == other.q
        result *= self.m == other.m
        result *= self.N == other.N
        result *= np.isclose(self.x, other.x).all()
        result *= np.isclose(self.v, other.v).all()
        result *= self.name == other.name
        return result

if __name__=="__main__":
    pass
