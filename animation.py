import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np


def animation(S):
    fig, (charge_axes, field_axes, phase_axes) = plt.subplots(3, squeeze=True, figsize=(10, 5))

    iteration = charge_axes.text(0.1, 0.9, 'i=x', horizontalalignment='center',
                                 verticalalignment='center', transform=charge_axes.transAxes)

    phase_plot, = phase_axes.plot([], [], "b.")
    phase_axes.set_xlim(0, S.L)
    maxv = 5 * np.mean(np.abs(S.particle_velocities))
    phase_axes.set_ylim(-maxv, maxv)
    phase_axes.set_xlabel("x")
    phase_axes.set_ylabel("v_x")
    phase_axes.vlines(S.x, -1, 1)

    charge_plot, = charge_axes.plot([], [])
    charge_axes.vlines(S.x, -1, 1)
    field_axes.vlines(S.x, -1, 1)
    field_plot, = field_axes.plot([], [])
    field_axes.set_ylabel(r"Field $E$")
    field_axes.set_xlim(0, S.L)
    charge_axes.set_xlim(0, S.L)
    charge_axes.set_ylabel(r"Charge density $\rho$")
    maxcharge = np.max(np.abs(S.charge_density))
    charge_axes.set_ylim(-maxcharge, maxcharge)
    maxfield = np.max(np.abs(S.electric_field))
    field_axes.set_ylim(-maxfield, maxfield)
    field_axes.set_ylabel(r"Field $E$")
    field_axes.set_xlim(0, S.L)
    charge_axes.set_xlim(0, S.L)
    charge_axes.set_ylabel(r"Charge density $\rho$, potential $V$")

    # fig.subplots_adjust(hspace=0)
    # potential_plot, = charge_axes.plot(S.x, S.potential, "g-")
    # charge_grid_scatter = charge_axes.scatter(S.x, np.zeros_like(S.x))
    # position_histogram = position_hist_axes.hist(x_particles,NG, alpha=0.1)
    # position_hist_axes.set_ylabel("$N$ at $x$")
    # position_hist_axes.set_xlim(0,L)
    # velocity_hist = velocity_hist_axes.hist(np.abs(v_particles),100)
    # velocity_hist_axes.set_xlabel("$x$")
    # velocity_hist_axes.set_xlabel("$v$")
    # velocity_hist_axes.set_ylabel("$N$ at $v$")
    # phase_axes_scatter = phase_axes.scatter([], [])


    def init():
        iteration.set_text("Iteration: ")
        charge_plot.set_data([], [])
        field_plot.set_data([], [])
        phase_plot.set_data([], [])
        # phase_axes_scatter.set_array([], [])
        # phase_axes.set_xlim(0,S.L)
        return [charge_plot, field_plot, phase_plot, iteration]  # phase_axes,

    def animate(i):
        charge_plot.set_data(S.x, S.charge_density[i])
        phase_plot.set_data(S.particle_positions[i], S.particle_velocities[i])
        field_plot.set_data(S.x, S.electric_field[i])
        # position_histogram.set_data(S.x_particles, NG)
        # phase_axes_scatter.set_array(S.particle_positions[i], S.particle_velocities[i])
        # iteration.set_text(i)
        iteration.set_text("Iteration: {}".format(i))
        return [charge_plot, field_plot, phase_plot, iteration]

    animation_object = anim.FuncAnimation(fig, animate, interval=100, frames=int(S.NT), blit=True, init_func=init)
    animation_object.save("video.mp4", fps=15, extra_args=['-vcodec', 'libx264'])
    plt.show()
    # return animation_object
