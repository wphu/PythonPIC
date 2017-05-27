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
        for plot in self.plots:
            plot.set_data([], [])

    def update(self, i):
        pass

    def return_animated(self):
        return self.plots


class FrequencyPlot(Plot):
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
    return {"x": species.position_history,
            "v_x": species.velocity_history[:, :, 0],
            "v_y": species.velocity_history[:, :, 1],
            "v_z": species.velocity_history[:, :, 2],
            }


class PhasePlot(Plot):
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


class VelocityDistributionPlot(Plot):
    def __init__(self, S, ax, v1="v_x"):
        super().__init__(S, ax)
        self.bin_arrays = []
        self.values = [phaseplot_values(species)[v1] for species in S.list_species]
        if len(self.values):
            maxxs = max([np.max(np.abs(v)) for v in self.values])
            self.ax.set_xlim(-maxxs, maxxs)
        for i, s, v in zip(range(len(S.list_species)), S.list_species, self.values):
            bin_array = np.linspace(v.min(), v.max())
            self.bin_arrays.append(bin_array)
            self.plots.append(
                ax.plot(*velocity_histogram_data(v[0], bin_array), colors[i])[0])
        self.ax.set_xlabel(rf"${v1}$")
        self.ax.set_ylabel(r"Number of particles")

        if len(self.bin_arrays):
            self.ax.set_xlim(min([bin_array.min() for bin_array in self.bin_arrays]),
                             max([bin_array.max() for bin_array in self.bin_arrays]))

    def update(self, i):
        for species, histogram, bin_array, v in zip(self.S.list_species, self.plots, self.bin_arrays, self.values):
            index = helper_functions.convert_global_to_particle_iter(i, species.save_every_n_iterations)
            histogram.set_data(*velocity_histogram_data(v[index], bin_array))


def velocity_histogram_data(arr, bins):
    """

    :param arr: particle velocity array
    :param bins: number of bins or array of edges 
    :return: x, y data on bins for linear plots
    """
    bin_height, bin_edge = np.histogram(arr, bins=bins)
    bin_center = (bin_edge[:-1] + bin_edge[1:]) * 0.5
    return bin_center, bin_height


class IterationCounter:
    def __init__(self, S, ax):
        self.S = S
        self.ax = ax
        self.counter = ax.text(0.1, 0.9, 'i=x', horizontalalignment='left',
                               verticalalignment='center', transform=ax.transAxes)

    def animation_init(self):
        self.counter.set_text("Iteration: ")

    def update(self, i):
        self.counter.set_text(f"Iteration: {i}/{self.S.NT}\nTime: {i*self.S.dt:.3g}/{self.S.NT*self.S.dt:.3g}")

    def return_animated(self):
        return [self.counter]


class FieldPlot(Plot):
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
                               self.S.grid.magnetic_field_history[i, :, self.j - 1])  # TODO remove -1 via ProcGrid


class CurrentPlot(Plot):
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
    def __init__(self, S, axes):
        plots = [FieldPlot(S, ax, j) for j, ax in enumerate(axes)]
        super().__init__(axes, plots)


class TripleCurrentPlot(PlotSet):
    def __init__(self, S, axes):
        plots = [CurrentPlot(S, ax, j) for j, ax in enumerate(axes)]
        super().__init__(axes, plots)


def animation(S, save: bool = False, alpha=1, frame_to_draw="animation"):
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
                        wspace=.25, hspace=0.6)  # TODO: remove particle windows if there are no particles

    phase_plot = PhasePlot(S, phase_axes, "x", "v_x", alpha)
    velocity_histogram = VelocityDistributionPlot(S, distribution_axes)
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
               *iteration.return_animated()]  # TODO: optimize this

    def animate(i):
        """draws the i-th frame of the simulation"""
        for plot in plots:
            plot.update(i)
        return results

    if frame_to_draw == "animation":
        def init():
            """initializes animation window for faster drawing"""
            for plot in plots:
                plot.animation_init()
            return results


        # noinspection PyTypeChecker
        animation_object = anim.FuncAnimation(fig, animate, interval=100,
                                              frames=np.arange(0, S.NT, helper_functions.calculate_particle_iter_step(S.NT),
                                                               dtype=int),
                                              blit=True, init_func=init)
        if save:
            helper_functions.make_sure_path_exists(S.filename)
            videofile_name = S.filename.replace(".hdf5", ".mp4")
            print(f"Saving animation to {videofile_name}")
            animation_object.save(videofile_name, fps=15, writer='ffmpeg', extra_args=['-vcodec', 'libx264'])
            print(f"Saved animation to {videofile_name}")
        return animation_object
    # elif isinstance(list) TODO: allow drawing snapshots
    elif isinstance(frame_to_draw, int):
        animate(iteration)
        if save:
            helper_functions.make_sure_path_exists(S.filename)
            file_name = S.filename.replace(".hdf5", ".png")
            print(f"Saving iteration {iteration} to {file_name}")
            fig.savefig(file_name)
        return fig
    else:
        raise ValueError("Incorrect frame_to_draw - must be 'animation' or number of iteration to draw.")