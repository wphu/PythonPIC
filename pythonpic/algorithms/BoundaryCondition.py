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
    species.x[~species.alive] = np.nan
    species.v[~species.alive] = np.nan


def apply_bc_buneman(grid, i, bc_function):
    grid.electric_field[0, 1] = bc_function(i * grid.dt, *grid.bc_params)
    # self.magnetic_field[0, :] = self.bc_function(i * self.dt, *self.bc_params) / self.c

class Laser:
    """
    Represents a boundary condition for field entering the simulation.

    `laser_wave` returns the sinusoidal field.
    `laser_envelope` returns the exponential envelope.
    `laser_pulse` returns the combination of the two.

    Examples
    ---------
    >>> Laser(0.5, 1).laser_envelope(0)
    1.0
    >>> Laser(0.5, 1).laser_wave(0)
    0.0
    >>> Laser(0.5, 1).laser_pulse(0)
    0.0
    >>> Laser(0.5, 1).laser_omega / 2 / np.pi
    1.0
    >>> np.isclose(Laser(1/2, 1).laser_wave(1), 0)
    True
    >>> np.isclose(Laser(1/2, 1).laser_pulse(1), 0)
    True
    >>> Laser(0.5, 1, 10).laser_envelope(10)
    1.0

    Parameters
    ----------
    laser_intensity : float
        Laser intensity in W/m^2
    laser_wavelength : float
        Laser wavelength in m
    envelope_center_t : float
        Center time for envelope
    envelope_width : float
        Envelope width.
    envelope_power : float
        Exponent for calculation of the pulse's shape.
    laser_phase : float
        Initial wavelength phase, in radians
    c : float
        Speed of light, in m/s
    epsilon_0 : float
        The physical constant
    """
    def __init__(self, laser_intensity, laser_wavelength, envelope_center_t=0, envelope_width=1, envelope_power=2, laser_phase = 0, c=1, epsilon_0=1):
        self.laser_wavelength = laser_wavelength
        self.laser_phase = laser_phase
        self.laser_omega = 2 * np.pi * c / laser_wavelength

        self.envelope_center_t = envelope_center_t
        self.envelope_width = envelope_width
        self.envelope_power = envelope_power
        self.laser_intensity = laser_intensity
        self.laser_amplitude = ((laser_intensity * 2 ) / (c * epsilon_0)) ** 0.5

    def wave_func(self, t):
        return np.sin(self.laser_omega * t + self.laser_phase)

    def laser_wave(self, t):
        return self.laser_amplitude * self.wave_func(t)

    def envelope_func(self, t):
        return np.exp(-(t - self.envelope_center_t) ** self.envelope_power / self.envelope_width)

    def laser_envelope(self, t):
        return self.laser_amplitude * self.envelope_func(t)

    def laser_pulse(self, t):
        return self.laser_wave(t) * self.envelope_func(t)

def non_periodic_bc(wrapped_function):
    return BoundaryCondition(kill_particles_outside_bounds, wrapped_function)


PeriodicBC = BoundaryCondition(return_particles_to_bounds, lambda x: None)
# laser = Laser(1, 10, 3)
# LaserBC = non_periodic_bc(laser.laser_pulse)
# WaveBC = non_periodic_bc(laser.laser_wave)
# EnvelopeBC = non_periodic_bc(laser.laser_envelope)
