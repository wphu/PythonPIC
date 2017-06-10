# coding=utf-8

import numpy as np


def return_particles_to_bounds(species):
    species.x %= species.grid.L


def kill_particles_outside_bounds(species):
    alive = (0 <= species.x) & (species.x < species.grid.L)
    if species.N_alive:
        species.x = species.x[alive]
        species.v = species.v[alive]
    species.N_alive = alive.sum()


class Laser:
    """
    Represents a boundary condition for field entering the simulation.

    `laser_wave` returns the sinusoidal field.
    `laser_envelope` returns the exponential envelope.
    `laser_pulse` returns the combination of the two.

    Examples
    ---------
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
    def __init__(self, laser_intensity,
                 laser_wavelength,
                 envelope_center_t=1,
                 envelope_width=1,
                 envelope_power=2,
                 laser_phase = 0,
                 c=1,
                 epsilon_0=1,
                 bc_function = "pulse"):
        self.laser_wavelength = laser_wavelength
        self.laser_phase = laser_phase
        self.laser_omega = 2 * np.pi * c / laser_wavelength

        self.envelope_center_t = envelope_center_t
        self.envelope_width = envelope_width
        self.envelope_power = envelope_power
        self.laser_intensity = laser_intensity
        # import ipdb; ipdb.set_trace()
        wave_impedance = 1/ (epsilon_0 * c)
        self.laser_amplitude = np.sqrt(self.laser_intensity * wave_impedance)
        t_12 = envelope_center_t
        self._taui = 0.5 / np.log(2)**(1/envelope_power) * t_12
        self._tau = 2**(1/envelope_power) * self._taui
        self._t_0 = self._tau * 10**(1/envelope_power)
        if bc_function == "pulse":
            self.bc_function = self.laser_pulse
        elif bc_function == "wave":
            self.bc_function = self.laser_wave
        elif bc_function == "envelope":
            self.bc_function = self.laser_envelope

    def wave_func(self, t):
        return np.sin(self.laser_omega * t + self.laser_phase)

    def laser_wave(self, t):
        return self.laser_amplitude * self.wave_func(t)

    def envelope_func(self, t):
        return np.exp(-0.5*((t - self._t_0) / self._taui)** self.envelope_power)

    def laser_envelope(self, t):
        return self.laser_amplitude * self.envelope_func(t)

    def laser_pulse(self, t):
        return self.laser_wave(t) * self.envelope_func(t)


