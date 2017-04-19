# coding=utf-8
import functools

class BoundaryCondition:
    def __init__(self, particle_bc, field_bc):
        self.particle_bc = particle_bc
        self.field_bc = field_bc

def return_particles_to_bounds(species, L):
    species.x %= L

def kill_particles_outside_bounds(species, L):
    species.alive = (0 < species.x) * (species.x < L)
    species.x[~species.alive] = -1 # TODO: replace with np.nan
    species.v[~species.alive] = 0 # replace with np.nan

def apply_bc_buneman(grid, i, bc_function):
    grid.electric_field[0, 1] = bc_function(i * grid.dt, *grid.bc_params)
    # self.magnetic_field[0, :] = self.bc_function(i * self.dt, *self.bc_params) / self.c

# TODO: figure out how to apply bc function
"""requirements for field boundary condition:
type of boundary condition (sin, exp, one times other)
place of application (left, right)
"""

PeriodicBC = BoundaryCondition(return_particles_to_bounds, lambda x: None)
# EnvelopeBC
# SineBC
LaserBC = BoundaryCondition(kill_particles_outside_bounds, functools.partial(apply_bc_buneman, lambda x: None)) # TODO: add bc function