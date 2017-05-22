"""Animates the simulation to show quantities that change over time"""
# coding=utf-8
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

from ..algorithms import helper_functions
from ..algorithms.helper_functions import colors, directions
from ..classes import simulation


# formatter = matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False)


def velocity_histogram_data(arr, bins):
    """

    :param arr: particle velocity array
    :param bins: number of bins or array of edges 
    :return: x, y data on bins for linear plots
    """
    bin_height, bin_edge = np.histogram(arr, bins=bins)
    bin_center = (bin_edge[:-1] + bin_edge[1:]) * 0.5
    return bin_center, bin_height


def animation(S, videofile_name=None, alpha=1):
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

    iteration = freq_axes.text(0.1, 0.9, 'i=x', horizontalalignment='left',
                               verticalalignment='center', transform=freq_axes.transAxes)

    fig.suptitle(str(S), fontsize=12)
    fig.subplots_adjust(top=0.81, bottom=0.08, left=0.15, right=0.95,
                        wspace=.25, hspace=0.6)  # TODO: remove particle windows if there are no particles

    current_plots = []
    electric_field_plots = []
    magnetic_field_plots = []

    charge_plot, = charge_axis.plot(S.grid.x, S.grid.charge_density_history[0, :], ".-", alpha=0.8)

    # TODO: 3d animation for transverse fields
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
    for j in range(3):
        current_plots.append(current_axes[j].plot(S.grid.x, S.grid.current_density_history[0, :, j], ".-",
                                                  alpha=0.9,
                                                  label=fr"$j_{directions[j]}$")[0])
        current_axes[j].set_xlim(0, S.grid.L)
        current_axes[j].set_ylabel(f"Current density $j_{directions[j]}$", color='b')
        current_axes[j].tick_params('y', colors='b')
        current_axes[j].set_xlabel(r"Position $x$")
        current_axes[j].ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)
        try:
            mincurrent = S.grid.current_density_history.min()
            maxcurrent = S.grid.current_density_history.max()
            current_axes[j].set_ylim(mincurrent, maxcurrent)
        except ValueError:
            pass
        current_axes[j].grid()
        current_axes[j].legend(loc='lower left')

        field_axes = current_axes[j].twinx()
        field_axes.set_xlim(0, S.grid.L)
        electric_field_plots.append(
            field_axes.plot(S.grid.x, S.grid.electric_field_history[0, :, j], "k.-", alpha=0.7,
                            label=f"$E_{directions[j]}$")[0])

        if j > 0:
            magnetic_field_plots.append(
                field_axes.plot(S.grid.x, S.grid.magnetic_field_history[0, :, j - 1], "m.-", alpha=0.7,
                                label=f"$B_{directions[j]}$")[0])
        field_axes.set_ylabel(r"Fields $E$, $B$", color='k')
        field_axes.tick_params('y', colors='k')
        field_axes.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)
        max_e = np.max(np.abs(S.grid.electric_field_history))
        max_b = np.max(np.abs(S.grid.magnetic_field_history))
        maxfield = max([max_e, max_b])
        field_axes.set_ylim(-maxfield, maxfield)
        field_axes.grid()
        field_axes.legend(loc='upper right')

    phase_dots = {}
    for i, species in enumerate(S.list_species):
        phase_dots[species.name], = phase_axes.plot([], [], colors[i] + ".", alpha=alpha)
    try:
        maxv = max([10 * np.mean(np.abs(species.velocity_history)) for species in S.list_species])
        phase_axes.set_ylim(-maxv, maxv)
    except ValueError:
        pass
    phase_axes.set_xlim(0, S.grid.L)
    phase_axes.yaxis.set_label_position("right")
    phase_axes.set_xlabel(r"Particle position $x$")
    phase_axes.set_ylabel(r"Particle velocity $v_x$")
    phase_axes.set_xticks(S.grid.x)
    phase_axes.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)
    phase_axes.grid()

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

    freq_plot, = freq_axes.plot([], [], "bo-", label="energy per mode")
    freq_axes.set_xlabel(r"Wavevector mode $k$")
    freq_axes.set_ylabel(r"Energy $E$")
    freq_axes.set_xlim(0, S.grid.NG / 2)
    freq_axes.set_ylim(S.grid.energy_per_mode_history.min(), S.grid.energy_per_mode_history.max())
    freq_axes.grid()
    freq_axes.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True, useOffset=False)
    freq_axes.yaxis.tick_right()
    freq_axes.yaxis.set_label_position("right")

    def init():
        """initializes animation window for faster drawing"""
        iteration.set_text("Iteration: ")
        freq_plot.set_data([], [])
        charge_plot.set_data([], [])
        for j in range(3):
            electric_field_plots[j].set_data([], [])
            if j > 0:
                magnetic_field_plots[j - 1].set_data([], [])
            for i, species, histogram in zip(range(len(S.list_species)), S.list_species, histograms):
                phase_dots[species.name].set_data([], [])
                histogram.set_data([], [])
            current_plots[j].set_data([], [])
        return [*current_plots, charge_plot, *electric_field_plots, *magnetic_field_plots, freq_plot,
                *phase_dots.values(), iteration]

    def animate(i):
        """draws the i-th frame of the simulation"""
        freq_plot.set_data(S.grid.k_plot, S.grid.energy_per_mode_history[i])
        iteration.set_text(f"Iteration: {i}/{S.NT}\nTime: {i*S.dt:.3g}/{S.NT*S.dt:.3g}")
        charge_plot.set_data(S.grid.x, S.grid.charge_density_history[i, :])
        for j in range(3):
            electric_field_plots[j].set_data(S.grid.x, S.grid.electric_field_history[i, :, j])
            if j > 0:
                magnetic_field_plots[j - 1].set_data(S.grid.x, S.grid.magnetic_field_history[i, :, j - 1])

            for i_species, species, histogram, bin_array in zip(range(len(S.list_species)), S.list_species, histograms,
                                                                bin_arrays):
                if helper_functions.is_this_saved_iteration(i, species.save_every_n_iterations):
                    index = helper_functions.convert_global_to_particle_iter(i, species.save_every_n_iterations)
                    phase_dots[species.name].set_data(species.position_history[index, :],
                                                      species.velocity_history[index, :, 0])
                    histogram.set_data(*velocity_histogram_data(species.velocity_history[index], bin_array))
            current_plots[j].set_data(S.grid.x, S.grid.current_density_history[i, :, j])

        return [*current_plots, charge_plot, *electric_field_plots, *magnetic_field_plots, freq_plot, *histograms,
                *phase_dots.values(),
                iteration]

    animation_object = anim.FuncAnimation(fig, animate, interval=100,
                                          frames=np.arange(0, S.NT, helper_functions.calculate_particle_iter_step(S.NT),
                                                           dtype=int),
                                          blit=True, init_func=init)
    if videofile_name:
        print(f"Saving animation to {videofile_name}")
        animation_object.save(videofile_name, fps=15, writer='ffmpeg', extra_args=['-vcodec', 'libx264'])
        print(f"Saved animation to {videofile_name}")
    return animation_object


if __name__ == "__main__":
    S = simulation.load_data("data_analysis/TS2/TS2.hdf5")
    anim = animation(S, "")
    plt.show()
