"""various helper functions"""
# coding=utf-8
import subprocess

import numpy as np

epsilon_0 = 1


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
