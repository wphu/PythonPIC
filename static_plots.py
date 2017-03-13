# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np


def static_plot_window(S, n, m):
    fig, axes = plt.subplots(n, m, figsize=(16, 10))
    fig.suptitle(S.date_ver_str, fontsize=12)  # TODO: add str(S)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    return fig, axes


def ESE_time_plots(S, axis):
    data = S.grid.energy_per_mode_history.T
    energies = [y for y in data]
    t = np.arange(S.NT) * S.dt
    for i, y in enumerate(energies):
        axis.plot(t, y, label=i)
    axis.legend()
    axis.grid()
    axis.set_xlabel("Time")
    axis.set_ylabel("Energy")
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
    axis.set_ylabel("Temperature")


def energy_time_plots(S, axis):
    for species in S.list_species:
        axis.plot(np.arange(S.NT) * S.dt, species.kinetic_energy_history, ".-",
                  label="Kinetic energy: {}".format(species.name))
    axis.plot(np.arange(S.NT) * S.dt, S.grid.grid_energy_history, ".-", label="Field energy (Fourier)",
              alpha=0.5)
    axis.plot(np.arange(S.NT) * S.dt, S.total_energy, ".-", label="Total energy")
    axis.plot(np.arange(S.NT) * S.dt, S.grid.epsilon_0 * (S.grid.electric_field_history ** 2).sum(axis=1) * 0.5,
                     ".-", label="Field energy (direct solve)", alpha=0.5)
    # TODO: implement direct field energy solver outside this place
    # TODO: why is direct field energy solver shifted
    axis.grid()
    axis.set_xlabel("Time")
    axis.set_ylabel("Energy")
    axis.legend(loc='lower right')


def velocity_distribution_plots(S, axis, i=0):
    for species in S.list_species:
        axis.hist(species.velocity_history[i, :, 0], bins=50, alpha=0.5)
    axis.set_title("Velocity distribution at iteration %d" % i)
    axis.grid()
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
        axis.plot(x, y)
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

if __name__ == "__main__":
    import Simulation

    S = Simulation.load_data("data_analysis/CO/COsimrun.hdf5")
    time_fig, axes = static_plot_window(S, 3, 2)

    ESE_time_plots(S, axes[0, 0])
    temperature_time_plot(S, axes[1, 0])
    energy_time_plots(S, axes[2, 0])
    phase_trajectories(S, axes[0, 1])
    velocity_distribution_plots(S, axes[1, 1])
    velocity_time_plots(S, axes[2, 1])

    plt.show()
