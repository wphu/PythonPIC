"""plotting suite for simulation data analysis, can be called from command line"""
# coding=utf-8
import argparse

import matplotlib.pyplot as plt

from ..classes import simulation
from ..visualization import animation
from ..visualization import static_plots

directory = "data_analysis/"


def plots(file, show: bool = True, save: bool = False, animate: bool = True, alpha: float = 1):
    """
    Runs visual analysis on saved hdf5 file. Currently runs:
    * energy vs time plot
    * electrostatic energy per mode vs time
    * temperature vs time
    * spectral analysis


    :param str file: hdf5 file location
    :param bool show: True if you want to show the plots right after creating
    :param bool save: True if you want to save the plots
    :param bool animate: True if you want to display/save animation
    :param float alpha: [0, 1] phaseplot dot opacity
    """
    if type(file) == simulation.Simulation:
        S = file
    else:
        print(f"Loading simulation data from {file}")
        S = simulation.load_data(file)
    static_plots.static_plots(S, S.filename.replace(".hdf5", ".png") if save else None)
    print(S)
    if animate:
        # noinspection PyUnusedLocal
        # this needs name due to matplotlib.animation
        anim = animation.animation(S, S.filename.replace(".hdf5", ".mp4") if save else None, alpha=alpha)
    if show:
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

    plots(args.filename, show=True, save=args.save)
