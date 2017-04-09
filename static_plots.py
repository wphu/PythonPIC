# coding=utf-8
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from helper_functions import directions, colors


def static_plot_window(S, N, M):
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(N, M)
    axes = [[fig.add_subplot(gs[n, m]) for m in range(M)] for n in range(N)]
    fig.suptitle(str(S), fontsize=12)

    # REFACTOR: separate window creation and axis layout into separate functions
    gs.update(left=0.075, right=0.95, bottom=0.075, top=0.8, hspace=0.45, wspace=0.025)  # , wspace=0.05, hspace=0.05
    return fig, axes


def ESE_time_plots(S, axis):
    data = S.grid.energy_per_mode_history

    # weights = (data ** 2).sum(axis=0) / (data ** 2).sum()
    #
    # # noinspection PyUnusedLocal
    # max_mode = weights.argmax()
    # # TODO: max_index = data[:, max_mode].argmax()

    t = np.arange(S.NT) * S.dt
    for i in range(1, 6):
        axis.plot(t, data[:, i], label=f"Mode {i}", alpha=0.8)
    for i in range(6, data.shape[1]):
        axis.plot(t, data[:, i], alpha=0.9)
    # axis.annotate(f"Mode {max_mode}",
    #               xy=(t[max_index], data[max_index, max_mode]),
    #               arrowprops=dict(facecolor='black', shrink=0.05),
    #               xytext=(t.mean(), data.max()/2))

    axis.legend(loc='upper right')
    axis.grid()
    axis.set_xlabel(f"Time [dt: {S.dt:.2e}]")
    axis.set_ylabel("Energy")
    axis.set_xlim(0, S.NT * S.dt)
    axis.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True, useOffset=False)
    axis.set_title("Energy per mode versus time")


def temperature_time_plot(S, axis, twinaxis=True):
    from matplotlib.font_manager import FontProperties
    fontP = FontProperties()
    fontP.set_size('small')

    t = np.arange(S.NT) * S.dt
    for species in S.list_species:
        meanv = species.velocity_history.mean(axis=1)
        meanv2 = (species.velocity_history ** 2).mean(axis=1)
        temperature = meanv2 - meanv ** 2
        temperature_parallel = temperature[:, 0]
        # TODO: temperature_transverse = temperature[:, 1:].sum(axis=1)
        axis.plot(t, temperature_parallel, label=species.name + r" $T_{||}$")
        if twinaxis:
            axis.plot(t, meanv2[:, 0], "--", label=species.name + r" $<v^2>$", alpha=0.5)
            axis.plot(t, meanv[:, 0] ** 2, "--", label=species.name + r" $<v>^2$", alpha=0.5)
    axis.legend(loc='center right', ncol=S.grid.n_species, prop=fontP)
    axis.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True, useOffset=False)
    axis.grid()
    axis.set_xlabel(r"Time $t$")
    axis.set_xlim(0, S.NT * S.dt)
    axis.set_ylabel("Temperature $t$")


def energy_time_plots(S, axis):
    for species in S.list_species:
        axis.plot(np.arange(S.NT) * S.dt, species.kinetic_energy_history, ".-",
                  label="Kinetic energy: {}".format(species.name), alpha=0.3)
    axis.plot(np.arange(S.NT) * S.dt, S.grid.grid_energy_history, ".-", label="Field energy (Fourier)",
              alpha=0.5)
    # axis.plot(np.arange(S.NT) * S.dt, S.grid.epsilon_0 * (S.grid.electric_field_history ** 2).sum(axis=1) * 0.5,
    #                  ".-", label="Field energy (direct solve)", alpha=0.5)
    # TODO: implement direct field energy solver outside this place
    # TODO: why is direct field energy solver shifted by a half? is this due to the staggered Leapfrog\Boris solver?
    axis.plot(np.arange(S.NT) * S.dt, S.total_energy, ".-", label="Total energy")
    axis.grid()
    axis.set_xlabel(r"Time $t$")
    axis.set_xlim(0, S.NT * S.dt)
    axis.set_ylabel(r"Energy $E$")
    axis.legend(loc='lower right')
    axis.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True, useOffset=False)


def velocity_distribution_plots(S, axis, i=0):
    for species in S.list_species:
        axis.hist(species.velocity_history[i, :, 0], bins=50, alpha=0.5, label=species.name)
    axis.set_title("Velocity distribution at iteration %d" % i)
    axis.grid()
    if S.grid.n_species > 1:
        axis.legend(loc='upper right')
    axis.set_xlabel(r"Velocity $v$")
    axis.set_ylabel(r"Number of superparticles")
    axis.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)


def phase_trajectories(S, axis, all=False):
    for species in S.list_species:
        if all:
            x = species.position_history[:, :]
            y = species.velocity_history[:, :, 0]
            axis.set_title("Phase space plot")
        else:
            i = int(species.N / 2)
            x = species.position_history[:, i]
            y = species.velocity_history[:, i, 0]
            axis.set_title("Phase space plot for particle {}".format(i))
        axis.plot(x, y, ".", label=species.name)
    axis.set_xlim(0, S.grid.L)
    if S.grid.n_species > 1:
        axis.legend()
    axis.set_xlabel(r"Position $x$")
    axis.set_ylabel(r"Velocity $v_x$")
    axis.grid()
    axis.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)


"""Multiple window plots"""


def velocity_time_plots(S, axis):
    t = np.arange(S.NT)*S.dt
    for s in S.list_species:
        for i in range(3):
            velocity = s.velocity_history[:, :, i]
            mean = velocity.mean(axis=1)
            std = velocity.std(axis=1)
            axis.plot(t, mean, "-", color=colors[i], label=f"{s.name} $v_{directions[i]}$", alpha=1)
            axis.fill_between(t, mean-std, mean+std, color=colors[i], alpha=0.3)
    axis.set_xlabel(r"Time $t$")
    axis.set_ylabel(r"Velocity $v$")
    if S.grid.n_species > 1:
        axis.legend()
    axis.grid()
    axis.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True, useOffset=False)


def static_plots(S, filename=False):
    if filename and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    time_fig, axes = static_plot_window(S, 3, 2)

    ESE_time_plots(S, axes[0][0])
    temperature_time_plot(S, axes[1][0])
    energy_time_plots(S, axes[2][0])
    velocity_time_plots(S, axes[0][1])
    axes[0][1].yaxis.tick_right()
    axes[0][1].yaxis.set_label_position("right")
    velocity_distribution_plots(S, axes[1][1])
    axes[1][1].yaxis.tick_right()
    axes[1][1].yaxis.set_label_position("right")
    velocity_distribution_plots(S, axes[2][1], S.NT - 1)
    axes[2][1].yaxis.tick_right()
    axes[2][1].yaxis.set_label_position("right")

    if filename:
        time_fig.savefig(filename)
    return time_fig


if __name__ == "__main__":
    import Simulation

    S = Simulation.load_data("data_analysis/TS2/TS2.hdf5")
    static_plots(S)
    S = Simulation.load_data("data_analysis/TS1/TS1.hdf5")
    static_plots(S)
    plt.show()
