import h5py
import numpy as np
import time

class Simulation(object):
    def __init__(self, NT, NGrid, NParticle, L=1, epsilon_0 = 1):
        self.x, self.dx = np.linspace(0,L,NGrid, retstep=True,endpoint=False)
        self.charge_density = np.zeros((NT, NGrid))
        self.electric_field = np.zeros((NT, NGrid))
        self.particle_positions = np.zeros((NT, NParticle))
        self.particle_velocities = np.zeros((NT, NParticle))
        self.NT, self.NGrid, self.NParticle = NT, NGrid, NParticle
        self.L, self.epsilon_0 = L, epsilon_0
    def update_grid(self, charge_density, electric_field):
        self.charge_density, self.electric_field = charge_density, electric_field
    def update_particles(self, i, particle_position, particle_velocity):
        self.particle_positions[i] = particle_position
        self.particle_velocities[i] = particle_velocity
    def fill_grid(self, charge_density, electric_field):
        self.charge_density, self.electric_field = charge_density, electric_field
    def fill_particles(self, particle_positions, particle_velocities):
        self.particle_positions = particle_positions
        self.particle_velocities = particle_velocities
    def save_data(self,
                        filename=time.strftime("%Y-%m-%d_%H-%M-%S.hdf5")):
        S = self
        with h5py.File(filename, "w") as f:
            f.create_dataset(name="Charge density", dtype=float, data=S.charge_density)
            f.create_dataset(name = "Electric field", dtype = float, data = S.electric_field)
            f.create_dataset(name = "Particle positions", dtype=float, data=S.particle_positions)
            f.create_dataset(name = "Particle velocities", dtype=float, data=S.particle_velocities)
            f.create_dataset(name = "Grid", dtype=float, data = S.x)
            f.attrs['NT'] = S.NT
            f.attrs['NGrid'] = S.NGrid
            f.attrs['NParticle'] = S.NParticle
        print("Saved file to {}".format(filename))
        return filename

def load_data(filename):
    with h5py.File(filename, "r") as f:
        charge_density=f['Charge density'][...]
        field = f['Electric field'][...]
        positions = f['Particle positions'][...]
        velocities = f['Particle velocities'][...]
        NT = f.attrs['NT']
        NGrid = f.attrs['NGrid']
        NParticle = f.attrs['NParticle']
    S = Simulation(NT, NGrid, NParticle)
    S.fill_grid(charge_density, field)
    S.fill_particles(positions, velocities)
    return S

if __name__=="__main__":
    NT = 25
    NGrid = 100
    NParticle = 10000

    charge_density_array = np.zeros((NT, NGrid))
    field_array = np.zeros_like(charge_density_array)
    positions_array = velocity_array = np.zeros((NT,NParticle))
    S = Simulation(NT, NGrid, NParticle)
    S.fill_grid(charge_density_array, field_array)
    S.fill_particles(positions_array, velocity_array)
    filename = S.save_data(filename="test.hdf5")

    S2 = load_data(filename)
    print(S2.charge_density, S2.electric_field, S2.particle_positions, S2.particle_velocities, S2.NT, S2.NGrid, S2.NParticle, sep='\n')
