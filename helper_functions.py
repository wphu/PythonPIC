"""various helper functions"""
# coding=utf-8
import argparse
import subprocess

import numpy as np

show_on_fail = False

def l2_norm(reference: np.ndarray, test: np.ndarray) -> float:
    """
    Calculates relative L-2 norm for accuracy testing

    :param np.ndarray reference: numpy array of values you assume to be correct
    :param np.ndarray test: numpy array of values you're attempting to test
    :return float: relative L-2 norm
    """
    # noinspection PyTypeChecker
    return np.sum((reference - test) ** 2) / np.sum(reference ** 2)


def l2_test(reference: np.ndarray, test: np.ndarray, rtol: float = 1e-3) -> bool:
    """
    Tests whether `reference` and `test` are close within relative tolerance level specified
    :param np.ndarray reference: numpy array of values you assume to be correct
    :param np.ndarray test: numpy array of values you're attempting to test
    :param float rtol: relative tolerance
    :return:
    """
    norm = l2_norm(reference, test)
    print("L2 norm: ", norm)
    return norm < rtol


def git_version() -> str:
    """
    :return: a short version of the git version hash
    """
    return subprocess.check_output(['git', 'describe', '--always']).decode()[:-1]


def plotting_parser(description):
    """
    Parses flags for showing or animating plots
    
    :param str description: Short program description
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--show", help="Show plots once the run finishes", action="store_true")
    parser.add_argument("--save", help="Save plots once the run finishes", action="store_true")
    parser.add_argument("--animate", help="Animate the run", action="store_true")
    args = parser.parse_args()
    show = args.show
    save = args.save
    animate = args.animate
    return show, save, animate


def get_dominant_mode(S):
    """
    Calculates the dominant mode from energy oscillations
    :param Simulation S: simulation object
    :type S: Simulation
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


colors = "brgyc"
directions = "xyz"