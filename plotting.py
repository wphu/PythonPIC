"""plotting suite for simulation data analysis, can be called from command line"""
# coding=utf-8
import argparse

import matplotlib.pyplot as plt

import Simulation
import animation
import static_plots
from spectrograph import spectral_analysis

directory = "data_analysis/"


def plotting(filename: str, show: bool = True, save: bool = False, animate: bool = True, lines: bool = False, alpha: float = 1):
    """
    Runs visual analysis on saved hdf5 file. Currently runs:
    * energy vs time plot
    * electrostatic energy per mode vs time
    * temperature vs time
    * spectral analysis


    :param str filename: hdf5 file location
    :param bool show: True if you want to show the plots right after creating
    :param bool save: True if you want to save the plots
    :param bool animate: True if you want to display/save animation
    :param bool lines: True if you want to draw trajectories on the phase plot
    :param float alpha: [0, 1] phaseplot dot opacity
    """
    print("Plotting for %s" % filename)
    S = Simulation.load_data(filename)
    static_plots.energy_time_plots(S, filename.replace(".hdf5", "_energy.png"))
    static_plots.ESE_time_plots(S, filename.replace(".hdf5", "_mode_energy.png"))
    static_plots.temperature_time_plot(S, filename.replace(".hdf5", "_temperature.png"))
    spectral_analysis(S, filename.replace(".hdf5", "_spectro.png"))
    if animate:
        if save:
            videofile_name = filename.replace(".hdf5", ".mp4")
        else:
            videofile_name = None
        # noinspection PyUnusedLocal
        anim = animation.animation(S, videofile_name, lines, alpha=alpha)  # this needs name due to matplotlib.animation
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

    plotting(args.filename, show=True, save=args.save, lines=args.lines)
