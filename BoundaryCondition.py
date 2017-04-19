# coding=utf-8

class BoundaryCondition:
    def __init__(self, particle_bc, field_bc = None):
        self.particle_bc = particle_bc
        # self.field_bc = field_bc

def return_particles_to_bounds(species, L):
    species.x %= L

def kill_particles_outside_bounds(species, L):
    species.alive = (0 < species.x) * (species.x < L)
    species.x[~species.alive] = -1 # TODO: replace with np.nan
    species.v[~species.alive] = 0 # replace with np.nan


PeriodicBC = BoundaryCondition(return_particles_to_bounds, )
LaserBC = BoundaryCondition(kill_particles_outside_bounds, )