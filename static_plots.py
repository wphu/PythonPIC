# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np


# TODO: a general static plot class and subclasses for each of these below - or use strategy functions
def ESE_time_plots(S, file_name):
    fig, axis = plt.subplots()
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
    fig.savefig(file_name)
    return fig


def temperature_time_plot(S, file_name):
    fig, axis = plt.subplots()
    t = np.arange(S.NT) * S.dt
    for species in S.all_species:
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
    fig.savefig(file_name)
    return fig


def energy_time_plots(S, file_name):
    fig2, energy_axes = plt.subplots()
    for species in S.all_species:
        energy_axes.plot(np.arange(S.NT) * S.dt, species.kinetic_energy_history, ".-",
                         label="Kinetic energy: {}".format(species.name))
    energy_axes.plot(np.arange(S.NT) * S.dt, S.grid.grid_energy_history, ".-", label="Field energy (Fourier)",
                     alpha=0.5)
    energy_axes.plot(np.arange(S.NT) * S.dt, S.total_energy, ".-", label="Total energy")
    energy_axes.plot(np.arange(S.NT) * S.dt, S.grid.epsilon_0 * (S.grid.electric_field_history ** 2).sum(axis=1) * 0.5,
                     ".-", label="Field energy (direct solve)", alpha=0.5)
    # TODO: implement direct field energy solver outside this place
    # TODO: why is direct field energy solver shifted

    energy_axes.set_title(S.date_ver_str)
    energy_axes.grid()
    energy_axes.set_xlabel("Time")
    energy_axes.set_ylabel("Energy")
    energy_axes.legend(loc='lower right')
    fig2.savefig(file_name)
    return fig2


def velocity_distribution_plots(S, file_name, i=0):
    fig, axis = plt.subplots()
    for species in S.all_species:
        axis.hist(S.velocity_history[species.name][i, :, 0], bins=50, alpha=0.5)
    axis.set_title("Velocity distribution at iteration %d" % i)
    axis.grid()
    axis.set_xlabel("v")
    axis.set_ylabel("N")
    return fig


def velocity_time_plots(s, dt):
    fig, axes = plt.subplots(3)
    labels = ["vx", "vy", "vz"]
    t = np.arange(s.NT) * dt
    for i in range(3):
        axes[i].plot(t, s.velocity_history[:, :, i])
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("t")
        axes[i].grid()
    return fig


def phase_trajectories(S, file_name, all=False):
    fig, axis = plt.subplots()
    assert S.all_species # has members
    for species in S.all_species:
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
    return fig


if __name__ == "__main__":
    import Simulation

    S = Simulation.load_data("data_analysis/HO1.hdf5")
    # for i in np.linspace(0, S.NT, 10, endpoint=False, dtype=int):
    phase_trajectories(S, "none.png")
    plt.show()
