"""plotting suite for simulation data analysis, can be called from command line"""
# coding=utf-8
import argparse

import matplotlib.pyplot as plt

from ..classes import simulation
from ..visualization import animation
from ..visualization import static_plots

directory = "data_analysis/"


def plots(file,
          show_static: bool = True,
          save_static: bool = False,
          show_animation: bool = True,
          save_animation: bool = True,
          alpha: float = 1):
    """
    Runs visual analysis on saved hdf5 file. Currently runs:
    * energy vs time plot
    * electrostatic energy per mode vs time
    * temperature vs time
    * spectral analysis
    Parameters
    ----------
    file : str or simulation.Simulation
    show : bool
        
    save : bool
    animate : bool
    alpha : float

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
    # else:
    #     raise ValueError("Passed arguments mean you wouldn't show or save anything.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="hdf5 file name for storing data")
    parser.add_argument('-save', action='store_false')
    parser.add_argument('-lines', action='store_true')
    args = parser.parse_args()
    if args.filename[-5:] != ".hdf5":
        args.filename += ".hdf5"

    plots(args.filename, show=True)
