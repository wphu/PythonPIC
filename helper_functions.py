"""various helper functions"""
# coding=utf-8
import subprocess
import argparse
import numpy as np


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
    data = S.grid.energy_per_mode_history
    weights = (data ** 2).sum(axis=0) / (data ** 2).sum()

    max_mode = weights.argmax()
    # max_index = data[:, max_mode].argmax()
    return max_mode