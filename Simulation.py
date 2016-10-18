import h5py
import numpy as np
import time
from helper_functions import date_version_string


class Simulation(object):
    """Contains data from one run of the simulation:
    NT: number of iterations
    NGrid: Number of points on the grid
    NParticle: Number of particles (one species right now)
    L: Length of the simulation domain
    epsilon_0: the physical constant
    particle_positions, velocities: shape (NT, NParticle) numpy arrays of historical particle data
    charge_density, electric_field: shape (NT, NGrid) numpy arrays of historical grid data
    """
    def __init__(self, NT, NGrid, NParticle, T, q=-1, m=1,
                 L=1, epsilon_0=1,
                 charge_density="empty",
                 electric_field="empty",
                 particle_positions="empty",
                 particle_velocities="empty",
                 kinetic_energy="empty",
                 field_energy="empty",
                 total_energy="empty",
                 date_ver_str="no date set"):
        self.x, self.dx = np.linspace(0, L, NGrid, retstep=True, endpoint=False)

        self.NT, self.NGrid, self.NParticle = NT, NGrid, NParticle

        def fill_array(hopefully_array, shape_tuple):
            if isinstance(hopefully_array, str):
                return np.zeros(shape_tuple)
            else:
                return hopefully_array

        self.charge_density = fill_array(charge_density, (NT, NGrid))
        self.electric_field = fill_array(electric_field, (NT, NGrid))
        self.particle_positions = fill_array(particle_positions, (NT, NParticle))
        self.particle_velocities = fill_array(particle_velocities, (NT, NParticle))
        self.kinetic_energy = fill_array(kinetic_energy, NT)
        self.field_energy = fill_array(field_energy, NT)
        self.total_energy = fill_array(total_energy, NT)
        self.L, self.epsilon_0, self.T = L, epsilon_0, T
        self.q, self.m = q, m
        self.date_ver_str = date_ver_str

    def update_grid(self, i, charge_density, electric_field):
        """Update the i-th set of field values"""
        self.charge_density[i], self.electric_field[i] = charge_density, electric_field

    def update_particles(self, i, particle_position, particle_velocity):
        """Update the i-th set of particle values"""
        self.particle_positions[i] = particle_position
        self.particle_velocities[i] = particle_velocity

    def update_diagnostics(self, i, diagnostics):
        kinetic_energy, field_energy, total_energy = diagnostics
        self.kinetic_energy[i] = kinetic_energy
        self.field_energy[i] = field_energy
        self.total_energy[i] = total_energy

    def fill_grid(self, charge_density, electric_field):
        self.charge_density, self.electric_field = charge_density, electric_field

    def fill_particles(self, particle_positions, particle_velocities):
        self.particle_positions = particle_positions
        self.particle_velocities = particle_velocities

    def fill_diagnostics(self, diagnostics):
        kinetic_energy, field_energy, total_energy = diagnostics
        self.kinetic_energy = kinetic_energy
        self.field_energy = field_energy
        self.total_energy = total_energy

    ######
    # data access
    ######

    def save_data(self, filename=time.strftime("%Y-%m-%d_%H-%M-%S.hdf5")):
        """Save simulation data to hdf5.
        filename by default is the timestamp for the simulation."""

        S = self
        with h5py.File(filename, "w") as f:
            f.create_dataset(name="Charge density", dtype=float, data=S.charge_density)
            f.create_dataset(name="Electric field", dtype=float, data=S.electric_field)
            f.create_dataset(name="Particle positions", dtype=float, data=S.particle_positions)
            f.create_dataset(name="Particle velocities", dtype=float, data=S.particle_velocities)
            f.create_dataset(name="Grid", dtype=float, data=S.x)
            f.create_dataset(name="Kinetic energy", dtype=float, data=S.kinetic_energy)
            f.create_dataset(name="Field energy", dtype=float, data=S.field_energy)
            f.create_dataset(name="Total energy", dtype=float, data=S.total_energy)
            f.attrs['NT'] = S.NT
            f.attrs['NGrid'] = S.NGrid
            f.attrs['NParticle'] = S.NParticle
            f.attrs['T'] = S.T
            f.attrs['L'] = S.L
            f.attrs['date_ver_str'] = date_version_string()
        print("Saved file to {}".format(filename))
        return filename


def load_data(filename):
    """Create a Simulation object from a hdf5 file"""
    with h5py.File(filename, "r") as f:
        charge_density = f['Charge density'][...]
        field = f['Electric field'][...]
        positions = f['Particle positions'][...]
        velocities = f['Particle velocities'][...]
        kinetic_energy = f['Kinetic energy'][...]
        field_energy = f['Field energy'][...]
        total_energy = f['Total energy'][...]
        NT = f.attrs['NT']
        T = f.attrs['T']
        L = f.attrs['L']
        NGrid = f.attrs['NGrid']
        NParticle = f.attrs['NParticle']
        date_ver_str = f.attrs['date_ver_str']
    S = Simulation(NT, NGrid, NParticle, T, charge_density=charge_density, electric_field=field, particle_positions=positions, particle_velocities=velocities, kinetic_energy=kinetic_energy,
                   field_energy=field_energy, total_energy=total_energy, L=L, date_ver_str=date_ver_str)
    S.fill_grid(charge_density, field)
    S.fill_particles(positions, velocities)
    S.fill_diagnostics((kinetic_energy, field_energy, total_energy))
    return S

if __name__ == "__main__":
    NT = 25
    NGrid = 100
    NParticle = 10000
    T = 1

    charge_density_array = np.zeros((NT, NGrid))
    field_array = np.zeros_like(charge_density_array)
    positions_array = velocity_array = np.zeros((NT, NParticle))
    S = Simulation(NT, NGrid, NParticle, T, charge_density=charge_density_array, electric_field=field_array,
                   particle_positions=positions_array, particle_velocities=velocity_array)
    filename = S.save_data(filename="test.hdf5")

    S2 = load_data(filename)
    print(S2.charge_density, S2.electric_field, S2.particle_positions, S2.particle_velocities, S2.NT, S2.T, S2.NGrid, S2.NParticle, sep='\n')
