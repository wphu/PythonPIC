"""various helper functions"""
import argparse
# coding=utf-8
import errno
import os
import subprocess
import warnings
from collections import namedtuple

import numpy as np

show_on_fail = True

def report_progress(i: int, NT: int):
    """
    Prints out a message on how many iterations out of how many total have been completed.

    Parameters
    ----------
    i : int
        Current iteration number
    NT : int
        Total iteration number

    Examples
    ----------
    >>> report_progress(0, 128)
    0/128 iterations (0%) done!
    >>> report_progress(33, 200)
    33/200 iterations (16%) done!
    >>> report_progress(200, 200)
    200/200 iterations (100%) done!

    """
    print(f"{i}/{NT} iterations ({i/NT*100:.0f}%) done!")


def git_version() -> str:
    """
    Returns the current git version identifier.
    -------
    str
        The current seven first characters of the current git version hash.
    """
    return subprocess.check_output(['git', 'describe', '--always']).decode()[:-1]


def calculate_particle_iter_step(NT, f=np.log2):
    """
    Calculate number of iterations between saving particle data.

    The function is meant to be easy to change.
    It should, however, rise slower than :math:`f(x) = x`.
    Good candidates are logarithms and roots.

    If the result is lower than 1, it returns 1.

    Parameters
    ----------
    NT : int
        total number of iterations
    f : function
        A function of a single variable returning a single variable

    Examples
    ----------
    >>> calculate_particle_iter_step(128, np.log2)
    7
    >>> calculate_particle_iter_step(128, np.sqrt)
    11
    >>> calculate_particle_iter_step(3, np.log10)
    1

    Returns
    -------
    int
        iteration step capped
    """
    result = int(f(NT))
    return result if result > 1 else 1


def calculate_particle_snapshots(NT, f = np.log2):
    """
    Calculates number of particle snapshots via `calculate_particle_iter_step`. See docs of that.

    Parameters
    ----------
    NT : int
        total number of iterations
    f : function
        A slowly rising function of a single variable returning a single variable

    Examples
    ----------
    >>> calculate_particle_snapshots(128, np.log2)
    19
    >>> calculate_particle_snapshots(128, np.sqrt)
    12
    >>> calculate_particle_snapshots(3, np.log10)
    4

    Returns
    -------
    int
        number of iteration steps to be saved.

    """
    return int(NT / calculate_particle_iter_step(NT, f)) + 1 # CHECK if result shouldn't be as NT, so remove + 1 here


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


def is_this_saved_iteration(i, save_every_n_iterations):
    return i % save_every_n_iterations == 0


def convert_global_to_particle_iter(i, save_every_n_iterations):
    return i // save_every_n_iterations


def plotting_parser(description):
    """
    Parses flags for showing or animating plots
    
    :param str description: Short program description
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--show-static", help="Show plots once the run finishes", action="store_true")
    parser.add_argument("--save-static", help="Save plots once the run finishes", action="store_true")
    parser.add_argument("--show-animation", help="Show the animation", action="store_true")
    parser.add_argument("--save-animation", help="Save the animation", action="store_true")
    args = parser.parse_args()
    return args.show_static, args.save_static, args.show_animation, args.save_animation


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

    max_mode = weights[1:].argmax() +1
    # max_index = data[:, max_mode].argmax()
    return max_mode


def did_it_thermalize(S):
    initial_velocities = np.array([s.velocity_history[0, :, 0].mean() for s in S.list_species])
    initial_velocity_stds = np.array([s.velocity_history[0, :, 0].std() for s in S.list_species])
    average_velocities = np.array([s.velocity_history[:, :, 0].mean() for s in S.list_species])
    return np.abs((initial_velocities - average_velocities)) > initial_velocity_stds

def make_sure_path_exists(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

colors = "brgyc"
directions = "xyz"
Constants = namedtuple('Constants', ['c', 'epsilon_0'])


def gamma_from_v(v, c):
    return 1 / np.sqrt(1 - ((v ** 2).sum(axis=1, keepdims=True)) / c ** 2)  # below eq 22 LPIC


def gamma_from_u(u, c):
    return np.sqrt(1 + ((u ** 2).sum(axis=1, keepdims=True) / c ** 2))


epsilon_zero = 8.854e-12 # F/m
electric_charge = 1.602e-19 # C
lightspeed = 2.998e8 #m /s
proton_mass = 1.6726219e-27 #kg
electron_rest_mass = 9.109e-31 # kg


def critical_density(wavelength):
    """
    Calculates the critical plasma density:
    .. math::
    n_c = m_e \varepsilon_0 * (\frac{2 \pi c}{e \lambda})^2

    Parameters
    ----------
    wavelength : in meters

    Returns
    -------

    """
    n_c = electron_rest_mass * epsilon_zero * ((2 * np.pi * lightspeed) / (electric_charge * wavelength)) ** 2
    return n_c


