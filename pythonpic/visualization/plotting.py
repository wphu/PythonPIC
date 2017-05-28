"""plotting suite for simulation data analysis, can be called from command line"""
# coding=utf-8
import argparse

import matplotlib.pyplot as plt

from ..classes import simulation
from ..visualization import animation
from ..visualization import static_plots

directory = "data_analysis/"


def plots(file,
          show_static: bool = False,
          save_static: bool = False,
          show_animation: bool = False,
          save_animation: bool = False,
          alpha: float = 1):
    """
    Wrapper to run visual analysis on saved hdf5 file. Displays static plots and animations.

    Parameters
    ----------
    file : str or simulation.Simulation
    show_static : bool
    save_static : bool
    show_animation : bool
    save_animation : bool
    alpha : float
        Used for opacity in plots

    Returns
    -------

    """
    if show_static or show_animation or save_animation or save_static:
        if type(file) == simulation.Simulation:
            S = file
        else:
            try:
                print(f"Loading simulation data from {file}")
                S = simulation.load_data(file)
            except:
                raise ValueError("Simulation file doesn't exist.")
        if save_static or show_static:
            static = static_plots.static_plots(S)
        if show_animation or save_animation:
            # noinspection PyUnusedLocal
            # this needs name due to matplotlib.animation
            anim = animation.animation(S, save_animation, alpha=alpha)
        if show_animation or show_static:
            plt.show()
        else:
            plt.clf()
            plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="hdf5 file name for storing data")
    parser.add_argument('-save', action='store_false')
    parser.add_argument('-lines', action='store_true')
    args = parser.parse_args()
    if args.filename[-5:] != ".hdf5":
        args.filename += ".hdf5"

    plots(args.filename, True, False, True, False)
