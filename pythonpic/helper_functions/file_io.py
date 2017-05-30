# coding=utf-8
import os

# from ..classes import load_simulation


def config_filename(run_name, category_name=None, version_number=None):
    """
    Prepares config filename for saving.

    Parameters
    ----------
    run_name : str

    category_name : str

    version_number : int


    Examples
    ----------
    >>> config_filename("run")
    'data_analysis/run/run.hdf5'
    >>> config_filename("run", "simulation_type")
    'data_analysis/simulation_type/run/run.hdf5'
    >>> config_filename("run", "simulation_type", 1)
    'data_analysis/simulation_type/v1/run/run.hdf5'

    Returns
    -------

    """
    return "data_analysis/"+\
           f"{category_name+'/' if category_name else ''}"+\
            f"{'v' + str(version_number) + '/' if version_number else ''}" +\
            f"{run_name}/{run_name}.hdf5"

#
