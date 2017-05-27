"""various helper functions"""
# coding=utf-8
import errno
import argparse
import subprocess
import warnings
import os
from collections import namedtuple

import numpy as np

show_on_fail = True


def l2_norm(reference: np.ndarray, test: np.ndarray) -> float:
    """
    Calculates relative L-2 norm for accuracy testing

    :param np.ndarray reference: numpy array of values you assume to be correct
    :param np.ndarray test: numpy array of values you're attempting to test
    :return float: relative L-2 norm
    """
    # noinspection PyTypeChecker
    return np.sum((reference - test) ** 2) / np.sum(reference ** 2)


def git_version() -> str:
    """
    :return: a short version of the git version hash
    """
    return subprocess.check_output(['git', 'describe', '--always']).decode()[:-1]


def calculate_particle_iter_step(NT):
    result = int(np.log2(NT))
    return result if result >1 else 1


def calculate_particle_snapshots(NT):
    return int(NT / calculate_particle_iter_step(NT)) + 1


def plasma_parameter(N_particles, N_grid, dx):
    return (N_particles / N_grid) * dx


def cold_plasma_frequency(electron_density, electron_mass=1, epsilon_0=1, electric_charge=1):
    return (electron_density * electric_charge ** 2 / electron_mass / epsilon_0) ** 0.5


def check_plasma_parameter(N_particles, N_grid, dx):
    pp = plasma_parameter(N_particles, N_grid, dx)
    if pp < 5:
        warnings.warn(f"Plasma parameter seems low at {pp:.3f}! Low density plasma.")
    else:
        print(f"Plasma parameter is {pp:.3f}, which seems okay.")


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
    data = S.grid.energy_per_mode_history
    weights = (data ** 2).sum(axis=0) / (data ** 2).sum()

    max_mode = weights.argmax()
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