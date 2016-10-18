import matplotlib.pyplot as plt
import numpy as np
from helper_functions import date_version_string

def all_the_plots(i):
    # x_particles = np.random.random(100)
    field_particles = electric_field_function(x_particles)
    fig, subplots = plt.subplots(3, 2, squeeze=True)
    (charge_axes, d1), (field_axes, d3), (position_hist_axes, velocity_hist_axes) = subplots
    fig.subplots_adjust(hspace=0)

    charge_axes.plot(x, charge_density, label="charge density")
    charge_axes.plot(x, potential, "g-")
    charge_axes.scatter(x, np.zeros_like(x))
    charge_axes.set_xlim(0, L)
    charge_axes.set_ylabel(r"Charge density $\rho$, potential $V$")

    position_hist_axes.hist(x_particles, NG, alpha=0.1)
    position_hist_axes.set_ylabel("$N$ at $x$")
    position_hist_axes.set_xlim(0, L)

    field_axes.set_ylabel(r"Field $E$")
    field_axes.scatter(x_particles, field_particles, label="interpolated field")
    field_axes.plot(x, electric_field, label="electric field")
    field_axes.set_xlim(0, L)

    velocity_hist_axes.set_xlabel("$x$")
    velocity_hist_axes.hist(np.abs(v_particles), 100)
    velocity_hist_axes.set_xlabel("$v$")
    velocity_hist_axes.set_ylabel("$N$ at $v$")
    d1.scatter(x_particles, v_particles)
    d1.set_xlim(0, L)
    plt.savefig("{:03d}.png".format(i))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    # plt.show()
    plt.clf()
    plt.close()


def energy_time_plots(S, videofile_name):
    fig2, energy_axes = plt.subplots()
    energy_axes.plot(np.arange(S.NT), (S.kinetic_energy), "o-", label="Kinetic energy")
    energy_axes.plot(np.arange(S.NT), (S.field_energy), "o-", label="Field energy")
    energy_axes.plot(np.arange(S.NT), (S.total_energy), "o-", label="Total energy")

    energy_axes.set_title(date_version_string())
    energy_axes.grid()
    energy_axes.set_xlabel("Time")
    energy_axes.set_ylabel("Energy")
    energy_axes.legend(loc='best')
    fig2.savefig(videofile_name)
    return fig2
