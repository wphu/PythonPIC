"""Animates the simulation to show quantities that change over time"""
# coding=utf-8
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

from ..algorithms import helper_functions
from ..algorithms.helper_functions import colors, directions
from ..classes import simulation, Species


# formatter = matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False)

def phase_plot(S, phase_axes, alpha):
    phase_dots = {}
    maxvs = []
    for i, species in enumerate(S.list_species):
        phase_dots[species.name], = phase_axes.plot([], [], colors[i] + ".", alpha=alpha)
        maxvs.append(max([10 * np.mean(np.abs(species.velocity_history)) for species in S.list_species]))
    if len(S.list_species) > 0:
        maxv = max(maxvs)
        phase_axes.set_ylim(-maxv, maxv)
    phase_axes.set_xlim(0, S.grid.L)
    phase_axes.yaxis.set_label_position("right")
    phase_axes.set_xlabel(r"Particle position $x$")
    phase_axes.set_ylabel(r"Particle velocity $v_x$")
    phase_axes.set_xticks(S.grid.x)
    phase_axes.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)
    phase_axes.grid()
    return phase_dots

def phase_plot_init(S, phase_dots):
    for species in S.list_species:
        phase_dots[species.name].set_data([], [])

def phase_plot_update(S, phase_dots, i):
    for species in S.list_species:
        if helper_functions.is_this_saved_iteration(i, species.save_every_n_iterations):
            index = helper_functions.convert_global_to_particle_iter(i, species.save_every_n_iterations)
            phase_dots[species.name].set_data(species.position_history[index, species.alive_history[index]],
                                          species.velocity_history[index, species.alive_history[index], 0])

def spatial_distribution_plot(S, charge_axis):
    charge_plot, = charge_axis.plot(S.grid.x, S.grid.charge_density_history[0, :], "-", alpha=0.8)
    charge_axis.set_xlim(0, S.grid.L)
    charge_axis.set_ylabel(f"Charge density $\\rho$")
    charge_axis.set_xticks(S.grid.x)
    charge_axis.set_xlabel(r"Position $x$")
    charge_axis.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)
    try:
        mincharge = np.min(S.grid.charge_density_history)
        maxcharge = np.max(S.grid.charge_density_history)
        charge_axis.set_ylim(mincharge, maxcharge)
    except ValueError:
        pass
    charge_axis.grid()
    return charge_plot

def spatial_distribution_plot_init(charge_plot):
    charge_plot.set_data([], [])

def spatial_distribution_plot_update(S, charge_plot, i):
    charge_plot.set_data(S.grid.x, S.grid.charge_density_history[i, :])

def velocity_distribution_plot(S, distribution_axes):
    histograms = []
    bin_arrays = []
    for i, s in enumerate(S.list_species):
        bin_array = np.linspace(s.velocity_history.min(), s.velocity_history.max())
        bin_arrays.append(bin_array)
        histograms.append(
            distribution_axes.plot(*velocity_histogram_data(s.velocity_history[0], bin_array), colors[i])[0])
    distribution_axes.grid()
    distribution_axes.yaxis.tick_right()
    distribution_axes.yaxis.set_label_position("right")
    distribution_axes.set_xlabel(r"Velocity $v$")
    distribution_axes.set_ylabel(r"Number of particles")
    distribution_axes.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)
    return histograms, bin_arrays

def velocity_distribution_init(histograms):
    for histogram in histograms:
        histogram.set_data([], [])

def velocity_distribution_update(S, histograms, bin_arrays, i):
    for species, histogram, bin_array in zip(S.list_species, histograms, bin_arrays):
        index = helper_functions.convert_global_to_particle_iter(i, species.save_every_n_iterations)
        histogram.set_data(*velocity_histogram_data(species.velocity_history[index], bin_array))

def velocity_histogram_data(arr, bins):
    """

    :param arr: particle velocity array
    :param bins: number of bins or array of edges 
    :return: x, y data on bins for linear plots
    """
    bin_height, bin_edge = np.histogram(arr, bins=bins)
    bin_center = (bin_edge[:-1] + bin_edge[1:]) * 0.5
    return bin_center, bin_height

def frequency_plot(S, freq_axes):
    freq_plot, = freq_axes.plot([], [], "bo-", label="energy per mode")
    freq_axes.set_xlabel(r"Wavevector mode $k$")
    freq_axes.set_ylabel(r"Energy $E$")
    freq_axes.set_xlim(0, S.grid.NG / 2)
    freq_axes.set_ylim(S.grid.energy_per_mode_history.min(), S.grid.energy_per_mode_history.max())
    freq_axes.grid()
    freq_axes.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True, useOffset=False)
    freq_axes.yaxis.tick_right()
    freq_axes.yaxis.set_label_position("right")
    return freq_plot

def frequency_plot_init(freq_plot):
    freq_plot.set_data([], [])

def frequency_plot_update(S, freq_plot, i):
    freq_plot.set_data(S.grid.k_plot, S.grid.energy_per_mode_history[i])

def iteration_counter(axes):
    iteration = axes.text(0.1, 0.9, 'i=x', horizontalalignment='left',
                               verticalalignment='center', transform=axes.transAxes)
    return iteration

def iteration_init(iteration):
    iteration.set_text("Iteration: ")

def iteration_update(S, iteration, i):
    iteration.set_text(f"Iteration: {i}/{S.NT}\nTime: {i*S.dt:.3g}/{S.NT*S.dt:.3g}")

def field_plots(S, field_axes, j):
    field_axes.set_xlim(0, S.grid.L)
    efield_plot, = field_axes.plot(S.grid.x, S.grid.electric_field_history[0, :, j], "k-", alpha=0.7,
                        label=f"$E_{directions[j]}$")

    bfield_plot, = field_axes.plot(S.grid.x, S.grid.magnetic_field_history[0, :, j-1], "m-", alpha=0.7,
                        label=f"$B_{directions[j-1]}$")
    field_axes.set_ylabel(r"Fields $E$, $B$", color='k')
    field_axes.tick_params('y', colors='k')
    field_axes.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)
    max_e = np.max(np.abs(S.grid.electric_field_history[:,:,j]))
    max_b = np.max(np.abs(S.grid.magnetic_field_history[:,:,j-1]))
    maxfield = max([max_e, max_b])
    field_axes.set_ylim(-maxfield, maxfield)
    field_axes.grid()
    field_axes.legend(loc='upper right')
    return efield_plot, bfield_plot

def field_plots_init(efield_plot, bfield_plot):
    efield_plot.set_data([], [])
    bfield_plot.set_data([], [])

def field_plots_update(S, efield_plot, bfield_plot, j, i):
    efield_plot.set_data(S.grid.x, S.grid.electric_field_history[i, :, j])
    bfield_plot.set_data(S.grid.x, S.grid.magnetic_field_history[i, :, j - 1])


def current_plot(S, current_axes, j):
    current_plot, = current_axes.plot(S.grid.x, S.grid.current_density_history[0, :, j], ".-",
                                              alpha=0.9,
                                              label=fr"$j_{directions[j]}$")
    current_axes.set_xlim(0, S.grid.L)
    current_axes.set_ylabel(f"Current density $j_{directions[j]}$", color='b')
    current_axes.tick_params('y', colors='b')
    current_axes.set_xlabel(r"Position $x$")
    current_axes.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)
    try:
        mincurrent = S.grid.current_density_history[:, :, j].min()
        maxcurrent = S.grid.current_density_history[:, :, j].max()
        current_axes.set_ylim(mincurrent, maxcurrent)
    except ValueError:
        pass
    current_axes.grid()
    current_axes.legend(loc='lower left')
    return current_plot

def current_plots_init(current_plot):
    current_plot.set_data([], [])

def current_plots_update(S, current_plot, j, i):
    current_plot.set_data(S.grid.x, S.grid.current_density_history[i, :, j])
# class ScatterPlot2D:
#     def __init__(self, ax, x, y, xlabel=None, ylabel=None):
#         self.xdata = x
#         self.ydata = y
#         self.plot, = ax.plot([], [], "")
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(ylabel)
#     def update(self, i):
#         self.plot.set_data(self.xdata[i], self.ydata[i])
#
#
# class PhasePlot(ScatterPlot2D):
#     def __init__(self, ax, s):
#         super().__init__(ax, s.position_history, s.velocity_history[:,:,0], r"$x$ position [m]", r"$v_x$ velocity [m/s]")



def animation(S, save: bool = False, alpha=1):
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

    iteration = iteration_counter(freq_axes)
    fig.suptitle(str(S), fontsize=12)
    fig.subplots_adjust(top=0.81, bottom=0.08, left=0.15, right=0.95,
                        wspace=.25, hspace=0.6)  # TODO: remove particle windows if there are no particles


    phase_dots = phase_plot(S, phase_axes, alpha)
    histograms, bin_arrays = velocity_distribution_plot(S, distribution_axes)
    freq_plot = frequency_plot(S, freq_axes)
    charge_plot = spatial_distribution_plot(S, charge_axis)

    current_plots = []
    electric_field_plots = []
    magnetic_field_plots = []
    # TODO: 3d animation for transverse fields
    for j in range(3):
        current_plots.append(current_plot(S, current_axes[j], j))
        field_axes = current_axes[j].twinx()
        efield_plot, bfield_plot = field_plots(S, field_axes, j)
        electric_field_plots.append(efield_plot)
        magnetic_field_plots.append(bfield_plot)



    def init():
        """initializes animation window for faster drawing"""
        iteration_init(iteration)
        frequency_plot_init(freq_plot)
        spatial_distribution_plot_init(charge_plot)
        phase_plot_init(S, phase_dots)
        velocity_distribution_init(histograms)
        for j in range(3):
            field_plots_init(electric_field_plots[j], magnetic_field_plots[j])
            current_plots_init(current_plots[j])
        return [*current_plots, charge_plot, *electric_field_plots, *magnetic_field_plots, freq_plot,
                *phase_dots.values(), iteration]

    def animate(i):
        """draws the i-th frame of the simulation"""
        frequency_plot_update(S, freq_plot, i)
        iteration_update(S, iteration, i)
        spatial_distribution_plot_update(S, charge_plot, i)
        phase_plot_update(S, phase_dots, i)
        velocity_distribution_update(S, histograms, bin_arrays, i)
        for j, efield_plot, bfield_plot in zip(range(3), electric_field_plots, magnetic_field_plots):
            field_plots_update(S, efield_plot, bfield_plot, j, i)
            current_plots_update(S, current_plots[j], j, i)

        return [*current_plots, charge_plot, *electric_field_plots, *magnetic_field_plots, freq_plot, *histograms,
                *phase_dots.values(), iteration]

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

