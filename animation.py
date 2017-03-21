"""Animates the simulation to show quantities that change over time"""
# coding=utf-8
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

colors = "brgyk"


def animation(S, videofile_name=None, lines=False, alpha=1):
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
    fig = plt.figure(figsize=(10, 10))
    charge_axes = fig.add_subplot(221)
    field_axes = fig.add_subplot(222)  # REFACTOR: merge charge_axes and field_axes using twiny
    # TODO: plot velocity distribution
    # TODO: add magnetic field
    phase_axes = fig.add_subplot(223)
    freq_axes = fig.add_subplot(224)

    iteration = freq_axes.text(0.1, 0.9, 'i=x', horizontalalignment='left',
                               verticalalignment='center', transform=freq_axes.transAxes)

    fig.suptitle(str(S), fontsize=12)
    fig.subplots_adjust(top=0.81)

    charge_plot, = charge_axes.plot([], [])
    charge_axes.set_xlim(0, S.grid.L)
    charge_axes.set_ylabel(r"Charge density $\rho$")
    mincharge = np.min(S.grid.charge_density_history)
    maxcharge = np.max(S.grid.charge_density_history)
    charge_axes.set_ylim(mincharge, maxcharge)
    charge_axes.set_xlim(0, S.grid.L)
    charge_axes.set_ylabel(r"Charge density $\rho$")
    # charge_axes.vlines(S.grid.x, mincharge/10, maxcharge/10)
    charge_axes.grid()

    field_plot, = field_axes.plot([], [])
    field_axes.set_ylabel(r"Electric field $E$")
    field_axes.set_xlim(0, S.grid.L)
    maxfield = np.max(np.abs(S.grid.electric_field_history))
    # field_axes.vlines(S.grid.x, -maxfield/10, maxfield/10)
    field_axes.grid()
    field_axes.set_ylim(-maxfield, maxfield)
    field_axes.set_ylabel(r"Field $E$")
    field_axes.set_xlim(0, S.grid.L)

    phase_dots = {}
    if lines:
        phase_lines = {}
    for i, species in enumerate(S.list_species):
        phase_dots[species.name], = phase_axes.plot([], [], colors[i] + ".", alpha=alpha)
        if lines:
            phase_lines[species.name], = phase_axes.plot([], [], colors[i] + "-", alpha=alpha / 2, lw=0.7)
    maxv = max([10 * np.mean(np.abs(species.velocity_history)) for species in S.list_species])
    phase_axes.set_xlim(0, S.grid.L)
    phase_axes.set_ylim(-maxv, maxv)
    phase_axes.set_xlabel("$x$")
    phase_axes.set_ylabel("$v_x$")
    # phase_axes.vlines(S.grid.x, -maxv/10, maxv/10)
    phase_axes.grid()

    freq_plot, = freq_axes.plot([], [], "bo-", label="energy per mode")
    freq_axes.set_xlabel("k")
    freq_axes.set_ylabel("E")
    freq_axes.set_xlim(0, S.grid.NG / 2)
    freq_axes.set_ylim(S.grid.energy_per_mode_history.min(), S.grid.energy_per_mode_history.max())
    freq_axes.grid()

    # fig.tight_layout()

    def init():
        """initializes animation window for faster drawing"""
        iteration.set_text("Iteration: ")
        charge_plot.set_data(S.grid.x, np.zeros_like(S.grid.x))
        field_plot.set_data(S.grid.x, np.zeros_like(S.grid.x))
        freq_plot.set_data(S.grid.k_plot, np.zeros_like(S.grid.k_plot))
        for species in S.list_species:
            phase_dots[species.name].set_data([], [])
            if lines:
                phase_lines[species.name].set_data([], [])
        if lines:
            return [charge_plot, field_plot, *phase_dots.values(), iteration, *phase_lines.values()]
        else:
            return [charge_plot, field_plot, freq_plot, *phase_dots.values(), iteration]

    def animate(i):
        """draws the i-th frame of the simulation"""
        charge_plot.set_ydata(S.grid.charge_density_history[i])
        field_plot.set_ydata(S.grid.electric_field_history[i])
        freq_plot.set_ydata(S.grid.energy_per_mode_history[i])
        for species in S.list_species:
            phase_dots[species.name].set_data(species.position_history[i, :], species.velocity_history[i, :, 0])
            if lines:
                phase_lines[species.name].set_data(species.position_history[:i + 1, ::10].T,
                                                   species.velocity_history[:i + 1, ::10, 0].T)
        iteration.set_text(f"Iteration: {i}/{S.NT}\nTime: {i*S.dt:.3g}/{S.NT*S.dt:.3g}")

        if lines:
            return [charge_plot, field_plot, freq_plot, *phase_dots.values(), iteration, *phase_lines.values()]
        else:
            return [charge_plot, field_plot, freq_plot, *phase_dots.values(), iteration]

    animation_object = anim.FuncAnimation(fig, animate, interval=100, frames=int(S.NT), blit=True, init_func=init)
    if videofile_name:
        print(f"Saving animation to {videofile_name}")
        animation_object.save(videofile_name, fps=15, writer='ffmpeg', extra_args=['-vcodec', 'libx264'])
        print(f"Saved animation to {videofile_name}")
    return animation_object
