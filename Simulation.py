import h5py
import numpy as np
import time
from helper_functions import date_version_string
from Grid import Grid
from Species import Species
from Constants import Constants

class Simulation():
    """Contains data from one run of the simulation:
    NT: number of iterations
    NGrid: Number of points on the grid
    NParticle: Number of particles (one species right now)
    L: Length of the simulation domain
    epsilon_0: the physical constant
    particle_positions, velocities: shape (NT, NParticle) numpy arrays of historical particle data
    charge_density, electric_field: shape (NT, NGrid) numpy arrays of historical grid data
    """
    def __init__(self, NT, dt, constants: Constants, grid: Grid, list_species):
        self.grid = grid
        self.all_species = list_species
        self.field_energy = np.zeros(NT)
        self.total_energy = np.zeros(NT)
        self.constants = constants
        self.epsilon_0, self.NT, self.dt = epsilon_0, NT, dt
        self.date_ver_str = date_version_string()
        # TODO: add more information about run, maybe to plotting

    def update_grid(self, i, grid = NotImplemented):
        """Update the i-th set of field values"""
        self.grid.save_field_values(i)

    def update_particles(self, i, list_species = NotImplemented):
        """Update the i-th set of particle values"""
        for species in self.all_species:
            species.save_particle_values(i)

    def update_diagnostics(self, i, kinetic_energy, field_energy, total_energy):
        self.kinetic_energy[i] = kinetic_energy
        self.field_energy[i] = field_energy
        self.total_energy[i] = total_energy

    def save_data(self, filename=time.strftime("%Y-%m-%d_%H-%M-%S.hdf5"), runtime=None):
        """Save simulation data to hdf5.
        filename by default is the timestamp for the simulation."""

        S = self
        with h5py.File(filename, "w") as f:
            grid_data = f.create_group('grid')
            self.grid.save_to_h5py(grid_data)

            all_species = f.create_group('species')
            for species in S.all_species:
                species_data = all_species.create_group(species.name)
                species.save_to_h5py(species_data)
            f.create_dataset(name="Field energy", dtype=float, data=S.field_energy)
            f.create_dataset(name="Total energy", dtype=float, data=S.total_energy)

            f.attrs['dt'] = S.dt
            f.attrs['NT'] = S.NT
            f.attrs['date_ver_str'] = date_version_string()
            if runtime:
                f.attrs['runtime'] = runtime
        print("Saved file to {}".format(filename))
        return filename

    def __eq__(self, other):
        result = True
        assert self.date_ver_str == other.date_ver_str, "date not equal!"
        result *= self.date_ver_str == other.date_ver_str

        assert self.epsilon_0 == other.epsilon_0, "epsilon 0 not equal!"
        result *= self.epsilon_0 == other.epsilon_0

        assert self.NT == other.NT, "NT not equal!"
        result *= self.NT == other.NT

        assert self.dt == other.dt, "NT not equal!"
        result *= self.dt == other.dt

        for this_species, other_species in zip(self.all_species, other.all_species):
            assert this_species == other_species, "{} and {} not equal!".format(this_species.name, other_species.name)
            result *= this_species == other_species
        assert self.grid == other.grid, "grid not equal!"
        result *= self.grid == other.grid
        return result


def read_hdf5_group(group):
    print("Group:", group)
    for i in group:
        print(i, group[i])
    for attr in group.attrs:
        print(attr, group.attrs[attr])


def load_data(filename):
    """Create a Simulation object from a hdf5 file"""
    with h5py.File(filename, "r") as f:
        total_energy = f['Total energy'][...]

        NT = f.attrs['NT']
        dt = f.attrs['dt']

        grid_data = f['grid']
        NG = grid_data.attrs['NGrid']
        grid = Grid(NT=NT, NG=NG)
        grid.load_from_h5py(grid_data)

        all_species = []
        for species_group_name in f['species']:
            # import ipdb; ipdb.set_trace()
            species_group = f['species'][species_group_name]
            species = Species(1,1,1, NT=NT)
            species.load_from_h5py(species_group)
            all_species.append(species)
        date_ver_str = f.attrs['date_ver_str']

    S = Simulation(NT, dt, grid.epsilon_0, grid, all_species, date_ver_str)

    S.total_energy = total_energy

    # for species in all_species:
    #     S.position_history[species.name], S.velocity_history[species.name], S.kinetic_energy_history[species.name] = particle_histories[species.name]
    return S
