# coding=utf-8
import warnings

import numpy as np


def plasma_parameter(N_particles, N_grid, dx):
    """
    Estimates the plasma parameter as the number of particles per step.

    Parameters
    ----------
    N_particles : int, float
        Number of physical particles
    N_grid : int
        Number of grid cells
    dx : float
        grid step size
    """
    return (N_particles / N_grid) * dx


def check_plasma_parameter(N_particles, N_grid, dx):
    pp = plasma_parameter(N_particles, N_grid, dx)
    if pp < 5:
        warnings.warn(f"Plasma parameter seems low at {pp:.3f}! Low density plasma.")
    else:
        print(f"Plasma parameter is {pp:.3f}, which seems okay.")


def cold_plasma_frequency(electron_density, electron_mass=1, epsilon_0=1, electric_charge=1):
    return (electron_density * electric_charge ** 2 / electron_mass / epsilon_0) ** 0.5


def check_pusher_stability(plasma_frequency, dt):
    if plasma_frequency * dt < 2:
        print(f"Pusher seems stable with dt * plasma frequency = {dt * plasma_frequency:.2e} < 2.")
    else:
        warnings.warn(f"dt {dt} too high relative to plasma frequency {plasma_frequency}! Pusher may be unstable!")


def calculate_number_timesteps(T, dt):
    return int(T / dt) + 1


def get_dominant_mode(S):
    """
    Calculates the dominant mode from energy oscillations
    :param Simulation S: simulation object
    :type S: Simulation.Simulation
    :return: number of dominant mode
    :rtype: int
    """
    S.postprocess()
    data = S.grid.energy_per_mode_history
    weights = (data ** 2).sum(axis=0) / (data ** 2).sum()

    max_mode = weights[1:].argmax()
    # max_index = data[:, max_mode].argmax()
    return max_mode


def did_it_thermalize(S):
    initial_velocities = np.array([s.velocity_history[0, :, 0].mean() for s in S.list_species])
    initial_velocity_stds = np.array([s.velocity_history[0, :, 0].std() for s in S.list_species])
    average_velocities = np.array([s.velocity_history[:, :, 0].mean() for s in S.list_species])
    return np.abs(initial_velocities - average_velocities) > initial_velocity_stds


def gamma_from_v(v, c):
    return 1 / np.sqrt(1 - ((v ** 2).sum(axis=1, keepdims=True)) / c ** 2)  # below eq 22 LPIC


def gamma_from_u(u, c):
    return np.sqrt(1 + ((u ** 2).sum(axis=1, keepdims=True) / c ** 2))


epsilon_zero = 8.854e-12  # F/m
electric_charge = 1.602e-19  # C
lightspeed = 2.998e8  # m /s
proton_mass = 1.6726219e-27  # kg
electron_rest_mass = 9.109e-31  # kg


def critical_density(wavelength):
    """
    Calculates the critical plasma density:
    .. math::
    n_c = m_e \varepsilon_0 * (\frac{2 \pi c}{e \lambda})^2

    Parameters
    ----------
    wavelength : in meters

    Examples
    ----------
    >>> critical_density(1)
    1115085555081946.6
    >>> critical_density(1115085555081946.6**0.5)
    1.0

    Returns
    -------

    """
    n_c = electron_rest_mass * epsilon_zero * ((2 * np.pi * lightspeed) / (electric_charge * wavelength)) ** 2
    return n_c
