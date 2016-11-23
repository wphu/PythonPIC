import matplotlib.pyplot as plt
import numpy as np


# def all_the_plots(i):
#     # x_particles = np.random.random(100)
#     field_particles = electric_field_function(x_particles)
#     fig, subplots = plt.subplots(3, 2, squeeze=True)
#     (charge_axes, d1), (field_axes, d3), (position_hist_axes, velocity_hist_axes) = subplots
#     fig.subplots_adjust(hspace=0)
#
#     charge_axes.plot(x, charge_density, label="charge density")
#     charge_axes.plot(x, potential, "g-")
#     charge_axes.scatter(x, np.zeros_like(x))
#     charge_axes.set_xlim(0, L)
#     charge_axes.set_ylabel(r"Charge density $\rho$, potential $V$")
#
#     position_hist_axes.hist(x_particles, NG, alpha=0.1)
#     position_hist_axes.set_ylabel("$N$ at $x$")
#     position_hist_axes.set_xlim(0, L)
#
#     field_axes.set_ylabel(r"Field $E$")
#     field_axes.scatter(x_particles, field_particles, label="interpolated field")
#     field_axes.plot(x, electric_field, label="electric field")
#     field_axes.set_xlim(0, L)
#
#     velocity_hist_axes.set_xlabel("$x$")
#     velocity_hist_axes.hist(np.abs(v_particles), 100)
#     velocity_hist_axes.set_xlabel("$v$")
#     velocity_hist_axes.set_ylabel("$N$ at $v$")
#     d1.scatter(x_particles, v_particles)
#     d1.set_xlim(0, L)
#     plt.savefig("{:03d}.png".format(i))
#     figManager = plt.get_current_fig_manager()
#     figManager.window.showMaximized()
#     # plt.show()
#     fig.clf()
#     fig.close()

def ESE_time_plots(S, file_name):
    fig, axis = plt.subplots()
    data = S.energy_per_mode.T
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
        velocity_vals = S.velocity_history[species.name]
        meanv = velocity_vals.mean(axis=1)
        meanv2 = (velocity_vals**2).mean(axis=1)
        temperature = meanv2 - meanv**2
        temperature_parallel = temperature[:,0]
        temperature_transverse = temperature[:,1:].sum(axis=1)
        axis.plot(t, temperature_parallel, label=species.name + r" $T_{||}$")
    axis.legend(loc='best')
    axis.grid()
    axis.set_xlabel("Time")
    axis.set_ylabel("Temperature")
    fig.savefig(file_name)
    return fig

def energy_time_plots(S, file_name):
    fig2, energy_axes = plt.subplots()
    for species in S.all_species:
        energy_axes.plot(np.arange(S.NT) * S.dt, (S.kinetic_energy_history[species.name]), "o-", label="Kinetic energy: {}".format(species.name))
    energy_axes.plot(np.arange(S.NT) * S.dt, (S.field_energy), "o-", label="Field energy")
    energy_axes.plot(np.arange(S.NT) * S.dt, (S.total_energy), "o-", label="Total energy")

    energy_axes.set_title(S.date_ver_str)
    energy_axes.grid()
    energy_axes.set_xlabel("Time")
    energy_axes.set_ylabel("Energy")
    energy_axes.legend(loc='best')
    fig2.savefig(file_name)
    return fig2

if __name__=="__main__":
    import Simulation
    S = Simulation.load_data("1default.hdf5")
    # print(S.energy_per_mode)
    temperature_time_plot(S, "none.png")
