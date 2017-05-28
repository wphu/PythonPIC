"""Animates the simulation to show quantities that change over time"""
# coding=utf-8
import itertools

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

from ..algorithms import helper_functions
from ..algorithms.helper_functions import colors, directions


# formatter = matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False)
class Plot:
    """
    A plot for visualization. Mainly an abstract class for overloading with interesting kinds of diagnostics.

    Parameters
    ----------
    S : Simulation
        A `Simulation` object to pull data from.
    ax : matplotlib axis
        An axis to draw on
    """

    def __init__(self, S, ax):
        self.S = S
        self.ax = ax
        self.plots = []

        self.ax.set_xlim(0, S.grid.L)
        ax.set_xlabel(r"Position $x$")
        self.ax.set_xticks(S.grid.x)
        self.ax.grid()
        self.ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True,
                                 useOffset=False)  # TODO axis=both?
        self.ax.xaxis.set_ticklabels([])
        self.ax.yaxis.tick_right()
        self.ax.yaxis.set_label_position("right")

    def animation_init(self):
        """
        Zeroes out all data in all lines of the plot. Useful for animation.
        """
        for plot in self.plots:
            plot.set_data([], [])

    def update(self, i):
        """
        Updates the plot with information from a particular iteration of the simulation.

        Parameters
        ----------
        i : int
            Iteration of the simulation
        """
        pass

    def return_animated(self):
        """
        Returns an iterable of all items that have changed. Useful for animation
        """
        return self.plots


class FrequencyPlot(Plot):
    """
    Plots the spatial Fourier transform of field energy versus wave number.
    """  # REFACTOR move the fourier analysis to PostProcessedGrid; describe the math here as well as there

    def __init__(self, S, ax):
        super().__init__(S, ax)
        self.plots.append(ax.plot([], [], "o-", label="energy per mode")[0])
        self.ax.set_xlabel(r"Wavevector mode $k$")
        self.ax.set_ylabel(r"Energy $E$")
        self.ax.set_xlim(0, S.grid.NG / 2)
        self.ax.set_xticks(S.grid.k_plot)
        self.ax.set_ylim(S.grid.energy_per_mode_history.min(), S.grid.energy_per_mode_history.max())

    def update(self, i):
        self.plots[0].set_data(self.S.grid.k_plot, self.S.grid.energy_per_mode_history[i])


def phaseplot_values(species):
    """
    A convenience function to get a dictionary of values, to allow generalization of the PhasePlot class.

    The keys you can pull for phase plots are `x`, `v_x`, `v_y` and `v_z`.

    Parameters
    ----------
    species : Species
        A species to draw data from

    Returns
    -------
    A dictionary of phase plot values.
    """
    return {"x":                           species.position_history,
            "v_x": species.velocity_history[:, :, 0],
            "v_y": species.velocity_history[:, :, 1],
            "v_z": species.velocity_history[:, :, 2],
            }


class PhasePlot(Plot):
    """
    Draws a phase plot.

    The keys you can pull for phase plots are `x`, `v_x`, `v_y` and `v_z`.

    Parameters
    ----------
    v1, v2 : str
        keys for the phase plot.
    alpha : float
        An opacity value between 0 and 1. Useful for neat phase plots displaying density.
    """

    def __init__(self, S, ax, v1, v2, alpha):
        super().__init__(S, ax)
        self.x = [phaseplot_values(species)[v1] for species in S.list_species]
        self.y = [phaseplot_values(species)[v2] for species in S.list_species]
        if len(self.y):
            maxys = max([np.max(np.abs(y)) for y in self.y])
            self.ax.set_ylim(-maxys, maxys)
        for i, species in enumerate(S.list_species):
            self.plots.append(ax.plot([], [], colors[i] + ".", alpha=alpha)[0])
        self.ax.yaxis.set_label_position("right")
        self.ax.set_xlabel(rf"${v1}$")
        self.ax.set_ylabel(rf"${v2}$")

    def update(self, i):
        for plot, species, x, y in zip(self.plots, self.S.list_species, self.x, self.y):
            if helper_functions.is_this_saved_iteration(i, species.save_every_n_iterations):
                index = helper_functions.convert_global_to_particle_iter(i, species.save_every_n_iterations)
                # print(y[index, species.alive_history[index]]) #TODO: get alive history to work here!
                plot.set_data(x[index],  # , species.alive_history[index]],
                              y[index])  # , species.alive_history[index]])


class SpatialDistributionPlot(Plot):
    """
    Draws charge density from the grid.
    """

    def __init__(self, S, ax):
        super().__init__(S, ax)
        self.plots.append(ax.plot([], [], "-", alpha=0.8)[0])
        ax.set_ylabel(f"Charge density $\\rho$")
        try:
            mincharge = np.min(S.grid.charge_density_history)
            maxcharge = np.max(S.grid.charge_density_history)
            ax.set_ylim(mincharge, maxcharge)
        except ValueError:
            pass

    def update(self, i):
        self.plots[0].set_data(self.S.grid.x, self.S.grid.charge_density_history[i, :])


class Histogram(Plot):
    """
    Draws a histogram of a given value from the phase plot dataset.

    The keys you can pull for phase plots are `x`, `v_x`, `v_y` and `v_z`.

    Parameters
    ----------
    v1 : str
        A key to phase plot values.
    n_bins: int
        Number of bins to draw.
    """

    def __init__(self, S, ax, v1: str, n_bins: int = 50):
        super().__init__(S, ax)
        self.bin_arrays = []
        self.values = [phaseplot_values(species)[v1] for species in S.list_species]
        if len(self.values):
            maxxs = max([np.max(np.abs(v)) for v in self.values])
            self.ax.set_xlim(-maxxs, maxxs)
        for i, s, v in zip(range(len(S.list_species)), S.list_species, self.values):
            bin_array = np.linspace(v.min(), v.max(), n_bins)
            self.bin_arrays.append(bin_array)
            self.plots.append(
                ax.plot(*calculate_histogram_data(v[0], bin_array), colors[i])[0])
        self.ax.set_xlabel(rf"${v1}$")
        self.ax.set_ylabel(r"Number of particles")

        if len(self.bin_arrays):
            self.ax.set_xlim(min([bin_array.min() for bin_array in self.bin_arrays]),
                             max([bin_array.max() for bin_array in self.bin_arrays]))

    def update(self, i):
        for species, histogram, bin_array, v in zip(self.S.list_species, self.plots, self.bin_arrays, self.values):
            index = helper_functions.convert_global_to_particle_iter(i, species.save_every_n_iterations)
            histogram.set_data(*calculate_histogram_data(v[index], bin_array))


def calculate_histogram_data(arr, bins):
    """
    Calculates histogram values, normalized to the number of particles.

    Parameters
    ----------
    arr : ndarray
        Values of a particle property, for example, velocity
    bins : ndarray
        Bin edges for the histogram.
    Returns
    -------
    bin_center : ndarray
        Centers of histogram bars (the x array for plotting)
    bin_height : ndarray
        Heights of histogram bars (the y array for plotting)
    """
    bin_height, bin_edge = np.histogram(arr, bins=bins)  # OPTIMIZE
    bin_center = (bin_edge[:-1] + bin_edge[1:]) * 0.5
    return bin_center, bin_height


class IterationCounter:
    """
    A little widget inserted on an axis, displaying the iteration number and current simulation time.
    """

    def __init__(self, S, ax):
        self.S = S
        self.ax = ax
        self.counter = ax.text(0.1, 0.9, 'i=x', horizontalalignment='left',
                               verticalalignment='center', transform=ax.transAxes)

    def animation_init(self):
        self.counter.set_text("Iteration: \nTime: ")

    def update(self, i):
        self.counter.set_text(f"Iteration: {i}/{self.S.NT}\nTime: {i*self.S.dt:.3g}/{self.S.NT*self.S.dt:.3g}")

    def return_animated(self):
        return [self.counter]


class FieldPlot(Plot):
    """
    Draws electric and magnetic fields from the grid in a given direction

    Parameters
    ----------
    j : int
        Direction as Cartesian index number. 0: x, 1: y, 2: z
    """

    def __init__(self, S, ax, j):
        super().__init__(S, ax)
        self.j = j
        self.plots.append(ax.plot([], [], "r-", label=f"$E_{directions[j]}$")[0])
        self.plots.append(ax.plot([], [], "b-", label=f"$B_{directions[j]}$")[0])
        ax.set_ylabel(r"Fields $E$, $B$")
        max_e = np.max(np.abs(S.grid.electric_field_history[:, :, j]))
        max_b = np.max(np.abs(S.grid.magnetic_field_history[:, :, j - 1]))  # TODO remove -1 via ProcessedGrid
        maxfield = max([max_e, max_b])
        ax.set_ylim(-maxfield, maxfield)
        ax.legend(loc='upper right')

    def update(self, i):
        self.plots[0].set_data(self.S.grid.x, self.S.grid.electric_field_history[i, :, self.j])
        self.plots[1].set_data(self.S.grid.x,
                               self.S.grid.magnetic_field_history[i, :, self.j - 1])  # TODO remove -1 via enlarging last dimension of MagFieldHistory


class CurrentPlot(Plot):
    """
    Draws currents from the grid in a given direction.

    Parameters
    ----------
    j : int
        Direction as Cartesian index number. 0: x, 1: y, 2: z
    """

    def __init__(self, S, ax, j):
        super().__init__(S, ax)
        self.j = j
        self.plots.append(ax.plot(S.grid.x, S.grid.current_density_history[0, :, j], ".-",
                                  alpha=0.9,
                                  label=fr"$j_{directions[j]}$")[0])
        ax.set_ylabel(f"Current density $j_{directions[j]}$", color='b')
        ax.tick_params('y', colors='b')
        ax.legend(loc='lower left')
        try:
            mincurrent = S.grid.current_density_history[:, :, j].min()
            maxcurrent = S.grid.current_density_history[:, :, j].max()
            ax.set_ylim(mincurrent, maxcurrent)
        except ValueError:
            pass

    def update(self, i):
        self.plots[0].set_data(self.S.grid.x, self.S.grid.current_density_history[i, :, self.j])


class PlotSet:
    """
    A single object representing a few different plots on different axes.
    Useful for plotting sets of directional values (fields, currents).

    Parameters
    ----------
    axes : list
        List of axes to use.
    list_plots :
        List of `Plot`s to update and return.
    """

    def __init__(self, axes, list_plots):
        self.axes = axes
        self.list_plots = list_plots

    def update(self, i):
        for plot in self.list_plots:
            plot.update(i)

    def animation_init(self):
        for plot in self.list_plots:
            plot.animation_init()

    def return_animated(self):
        return list(itertools.chain.from_iterable(plot.return_animated() for plot in self.list_plots))


class TripleFieldPlot(PlotSet):
    """
    Draws electric and magnetic field plots on the grid on a given list of axes.
    Parameters
    ----------
    S : Simulation
        Simulation to pull data from.
    axes : list
        List of matplotlib axes.
    """

    def __init__(self, S, axes: list):
        assert len(axes) <= 3, "Too many axes, we ran out of directions!"
        plots = [FieldPlot(S, ax, j) for j, ax in enumerate(axes)]
        super().__init__(axes, plots)


class TripleCurrentPlot(PlotSet):
    """
    Draws currents on the grid on a given list of axes.
    Parameters
    ----------
    S : Simulation
        Simulation to pull data from.
    axes : list
        List of matplotlib axes.
    """

    def __init__(self, S, axes: list):
        assert len(axes) <= 3, "Too many axes, we ran out of directions!"
        plots = [CurrentPlot(S, ax, j) for j, ax in enumerate(axes)]
        super().__init__(axes, plots)


def animation(S, save: bool = False, alpha=1, frame_to_draw="animation"):
    """

    Creates an animation from `Plot`s.

    Parameters
    ----------
    S : Simulation
        Data source
    save : bool
        Whether to save the simulation
    alpha : float
        Opacity value from 0 to 1, used in the phase plot.
    frame_to_draw : str, list or int
        Default value is "animation" - this causes the full animation to be played.
        A list such as [0, 10, 100, 500, 1000] causes the animation to plot iterations 0, 10, 100... etc
        and save them as pictures. Does not display the frames.

        An integer causes the animation to display a single iterations, optionally saving it as a picture.
    Returns
    -------
    figure or matplotlib animation
        Plot object, depending on `frame_to_draw`.
    """
    assert alpha <= 1, "alpha too large!"
    assert alpha >= 0, "alpha must be between 0 and 1!"
    """ animates the simulation, showing:
    * grid charge vs grid position
    * grid electric field vs grid position
    * particle phase plot (velocity vs position)
    * spatial energy modes

    S - Simulation object with run's data
    videofile_name - should be in format FILENAME.mp4;
        if not None, saves to file
    lines - boolean flag; draws particle trajectories on phase plot
    # TODO: investigate lines flag in animation
    alpha - float (0, 1) - controls opacity for phase plot

    returns: matplotlib figure with animation
    """
    fig = plt.figure(figsize=(13, 10))
    charge_axis = fig.add_subplot(421)
    current_axes = [fig.add_subplot(423 + 2 * i) for i in range(3)]
    distribution_axes = fig.add_subplot(424)
    phase_axes = fig.add_subplot(426)
    freq_axes = fig.add_subplot(428)

    fig.suptitle(str(S), fontsize=12)
    fig.subplots_adjust(top=0.81, bottom=0.08, left=0.15, right=0.95,
                        wspace=.25, hspace=0.6)  # REFACTOR: remove particle windows if there are no particles

    phase_plot = PhasePlot(S, phase_axes, "x", "v_x", alpha)
    velocity_histogram = Histogram(S, distribution_axes, "v_x")
    freq_plot = FrequencyPlot(S, freq_axes)
    charge_plot = SpatialDistributionPlot(S, charge_axis)
    iteration = IterationCounter(S, freq_axes)
    current_plots = TripleCurrentPlot(S, current_axes)
    field_plots = TripleFieldPlot(S, [current_ax.twinx() for current_ax in current_axes])

    plots = [phase_plot, velocity_histogram, freq_plot, charge_plot, iteration, current_plots, field_plots]
    results = [*field_plots.return_animated(),
               *current_plots.return_animated(),
               *charge_plot.return_animated(),
               *freq_plot.return_animated(),
               *velocity_histogram.return_animated(),
               *phase_plot.return_animated(),
               *iteration.return_animated()]  # REFACTOR: use itertools


    def animate(i, verbose=False):
        """draws the i-th frame of the simulation"""
        if verbose:
            helper_functions.report_progress(i, S.NG)
        for plot in plots:
            plot.update(i)
        return results

    if frame_to_draw == "animation":
        print("Drawing full animation.")
        def init():
            """initializes animation window for faster drawing"""
            for plot in plots:
                plot.animation_init()
            return results

        frames = np.arange(0, S.NT,
                           helper_functions.calculate_particle_iter_step(S.NT),
                           dtype=int)

        # noinspection PyTypeChecker
        animation_object = anim.FuncAnimation(fig, animate, interval=100,
                                              frames=frames,
                                              blit=True, init_func=init,
                                              fargs=True)
        if save:
            helper_functions.make_sure_path_exists(S.filename)
            videofile_name = S.filename.replace(".hdf5", ".mp4")
            print(f"Saving animation to {videofile_name}")
            animation_object.save(videofile_name, fps=15, writer='ffmpeg', extra_args=['-vcodec', 'libx264'])
            print(f"Saved animation to {videofile_name}")
        return animation_object
    elif isinstance(frame_to_draw, list):
        print("Drawing frames." + frame_to_draw)
        for i in frame_to_draw:
            animate(i)
            helper_functions.make_sure_path_exists(S.filename)
            file_name = S.filename.replace(".hdf5", f"{i}.png")
            print(f"Saving iteration {iteration} to {file_name}")
            fig.savefig(file_name)
        return fig
    elif isinstance(frame_to_draw, int):
        print("Drawing iteration", int)
        animate(iteration)
        if save:
            helper_functions.make_sure_path_exists(S.filename)
            file_name = S.filename.replace(".hdf5", ".png")
            print(f"Saving iteration {iteration} to {file_name}")
            fig.savefig(file_name)
        return fig
    else:
        raise ValueError("Incorrect frame_to_draw - must be 'animation' or number of iteration to draw.")
