# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


def static_plot_window(S, N, M):
    fig = plt.figure(figsize=(13, 8))
    gs = gridspec.GridSpec(N, M)
    axes = [[fig.add_subplot(gs[n,m]) for m in range(M)] for n in range(N)]
    fig.suptitle(str(S), fontsize=12)  # TODO: add str(S)
    gs.update(left = 0.05, right=0.95, bottom=0.075, top=0.8) # , wspace=0.05, hspace=0.05
    return fig, axes


def ESE_time_plots(S, axis):
    data = S.grid.energy_per_mode_history
    weights = (data**2).sum(axis=0) / (data**2).sum()
    # import ipdb; ipdb.set_trace()

    max_mode = weights.argmax()
    max_index = data[:, max_mode].argmax()

    t = np.arange(S.NT) * S.dt
    # for i, y in enumerate(energies):
    for i in range(5):
        axis.plot(t, data[:, i], label=f"Mode {i}", alpha=0.8)
    for i in range(5, data.shape[1]):
        axis.plot(t, data[:, i], alpha=0.9)
    # axis.annotate(f"Mode {max_mode}",
    #               xy=(t[max_index], data[max_index, max_mode]),
    #               arrowprops=dict(facecolor='black', shrink=0.05),
    #               xytext=(t.mean(), data.max()/2))

    axis.legend(loc='upper right')
    axis.grid()
    axis.set_xlabel(f"Time [dt: {S.dt:.3e}]")
    axis.set_ylabel("Energy")
    axis.set_xlim(0, S.NT * S.dt)
    axis.set_title("Energy per mode versus time")


def temperature_time_plot(S, axis):
    t = np.arange(S.NT) * S.dt
    for species in S.list_species:
        meanv = species.velocity_history.mean(axis=1)
        meanv2 = (species.velocity_history ** 2).mean(axis=1)
        temperature = meanv2 - meanv ** 2
        temperature_parallel = temperature[:, 0]
        temperature_transverse = temperature[:, 1:].sum(axis=1)
        axis.plot(t, temperature_parallel, label=species.name + r" $T_{||}$")
        axis.plot(t, meanv2[:, 0], label=species.name + r" $<v^2>$")
        axis.plot(t, meanv[:, 0] ** 2, label=species.name + r" $<v>^2$")
    axis.legend(loc='lower right')
    axis.grid()
    axis.set_xlabel("Time")
    axis.set_xlim(0, S.NT * S.dt)
    axis.set_ylabel("Temperature")


def energy_time_plots(S, axis):
    for species in S.list_species:
        axis.plot(np.arange(S.NT) * S.dt, species.kinetic_energy_history, ".-",
                  label="Kinetic energy: {}".format(species.name))
    axis.plot(np.arange(S.NT) * S.dt, S.grid.grid_energy_history, ".-", label="Field energy (Fourier)",
              alpha=0.5)
    # axis.plot(np.arange(S.NT) * S.dt, S.grid.epsilon_0 * (S.grid.electric_field_history ** 2).sum(axis=1) * 0.5,
    #                  ".-", label="Field energy (direct solve)", alpha=0.5)
    axis.plot(np.arange(S.NT) * S.dt, S.total_energy, ".-", label="Total energy")
    # TODO: implement direct field energy solver outside this place
    # TODO: why is direct field energy solver shifted
    axis.grid()
    axis.set_xlabel("Time")
    axis.set_xlim(0, S.NT * S.dt)
    axis.set_ylabel("Energy")
    axis.legend(loc='lower right')


def velocity_distribution_plots(S, axis, i=0):
    for species in S.list_species:
        axis.hist(species.velocity_history[i, :, 0], bins=50, alpha=0.5, label=species.name)
    axis.set_title("Velocity distribution at iteration %d" % i)
    axis.grid()
    axis.legend(loc='upper right')
    axis.set_xlabel("v")
    axis.set_ylabel("N")


def phase_trajectories(S, axis, all=False):
    assert S.list_species  # has members
    for species in S.list_species:
        # for i in range(species.N):
        if all:
            x = species.position_history[:, :]
            y = species.velocity_history[:, :, 0]
            axis.set_title("Phase space plot")
        else:
            i = int(species.N / 2)
            x = species.position_history[:, i]
            y = species.velocity_history[:, i, 0]
            axis.set_title("Phase space plot for particle {}".format(i))
        axis.plot(x, y, ".")
    axis.set_xlim(0, S.grid.L)
    # noinspection PyUnboundLocalVariable
    axis.set_xlabel("x")
    axis.set_ylabel("vx")
    axis.grid()


"""Multiple window plots"""


def velocity_time_plots(S, axis):
    labels = ["vx", "vy", "vz"]
    t = np.arange(S.NT)
    for s in S.list_species:
        for i in range(3):
            axis.plot(t, s.velocity_history[:, int(s.N / 2), i], label=s.name + labels[i])
    axis.set_xlabel("t")
    plt.legend()
    axis.grid()

def static_plots(S, filename=False):
    time_fig, axes = static_plot_window(S, 3, 2)

    ESE_time_plots(S, axes[0][0])
    temperature_time_plot(S, axes[1][0])
    energy_time_plots(S, axes[2][0])
    phase_trajectories(S, axes[0][1])
    velocity_distribution_plots(S, axes[1][1])
    velocity_distribution_plots(S, axes[2][1], S.NT-1)

    if filename:
            time_fig.savefig(filename)
    return time_fig


if __name__ == "__main__":
    import Simulation

    S = Simulation.load_data("data_analysis/CO/COsimrun.hdf5")
    static_plots(S)
    plt.show()
