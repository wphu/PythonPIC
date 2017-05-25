# coding=utf-8

import numpy as np


class BoundaryCondition:
    def __init__(self, particle_bc, field_bc):
        self.particle_bc = particle_bc
        self.field_bc = field_bc


def return_particles_to_bounds(species):
    species.x %= species.grid.L


def kill_particles_outside_bounds(species):
    species.alive = (0 < species.x) * (species.x < species.grid.L)
    species.x[~species.alive] = -1  # TODO: replace with np.nan
    species.v[~species.alive] = 0  # replace with np.nan


def apply_bc_buneman(grid, i, bc_function):
    grid.electric_field[0, 1] = bc_function(i * grid.dt, *grid.bc_params)
    # self.magnetic_field[0, :] = self.bc_function(i * self.dt, *self.bc_params) / self.c


class Laser:
    def __init__(self, laser_wavelength, envelope_center_t, envelope_width, envelope_power=2, c=1, laser_phase=0):
        self.laser_wavelength = laser_wavelength
        self.laser_phase = laser_phase
        self.laser_omega = 2 * np.pi * c / laser_wavelength

        self.envelope_center_t = envelope_center_t
        self.envelope_width = envelope_width
        self.envelope_power = envelope_power
        self.c = c

    def laser_wave(self, t):
        return np.sin(self.laser_omega * t + self.laser_phase)

    def laser_envelope(self, t):
        return np.exp(-(t - self.envelope_center_t) ** self.envelope_power / self.envelope_width)

    def laser_pulse(self, t):
        return self.laser_wave(t) * self.laser_envelope(t)


def non_periodic_bc(wrapped_function):
    return BoundaryCondition(kill_particles_outside_bounds, wrapped_function)


PeriodicBC = BoundaryCondition(return_particles_to_bounds, lambda x: None)
# laser = Laser(1, 10, 3)
# LaserBC = non_periodic_bc(laser.laser_pulse)
# WaveBC = non_periodic_bc(laser.laser_wave)
# EnvelopeBC = non_periodic_bc(laser.laser_envelope)
