import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

colors = "brgyk"

def animation(S, videofile_name, lines=False):
    fig = plt.figure()#(figsize=(10,15))
    charge_axes = fig.add_subplot(221)
    field_axes = fig.add_subplot(222)
    phase_axes = fig.add_subplot(223)
    freq_axes = fig.add_subplot(224)
    # fig, (charge_axes, field_axes, phase_axes, freq_axes) = plt.subplots(4, squeeze=True, figsize=(10, 5))

    iteration = freq_axes.text(0.1, 0.9, 'i=x', horizontalalignment='left',
                                 verticalalignment='center', transform=freq_axes.transAxes)

    charge_axes.set_title(S.date_ver_str)
    charge_plot, = charge_axes.plot([], [])
    charge_axes.set_xlim(0, S.grid.L)
    charge_axes.set_ylabel(r"Charge density $\rho$")
    mincharge = np.min(S.charge_density_history)
    maxcharge = np.max(S.charge_density_history)
    charge_axes.set_ylim(mincharge, maxcharge)
    charge_axes.set_xlim(0, S.grid.L)
    charge_axes.set_ylabel(r"Charge density $\rho$")
    charge_axes.vlines(S.grid.x, mincharge, maxcharge)

    field_axes.vlines(S.grid.x, -1, 1)
    field_plot, = field_axes.plot([], [])
    field_axes.set_ylabel(r"Field $E$")
    field_axes.set_xlim(0, S.grid.L)
    maxfield = np.max(np.abs(S.electric_field_history))
    field_axes.set_ylim(-maxfield, maxfield)
    field_axes.set_ylabel(r"Field $E$")
    field_axes.set_xlim(0, S.grid.L)

    phase_dots = {}
    if lines:
        phase_lines = {}
    for i, species in enumerate(S.all_species):
        phase_dots[species.name], = phase_axes.plot([], [], colors[i]+".", alpha=1)
        if lines:
            phase_lines[species.name], = phase_axes.plot([], [], colors[i]+"-", alpha=0.7, lw=0.7)
    maxv = max([5 * np.mean(np.abs(S.velocity_history[species.name])) for species in S.all_species])
    phase_axes.set_xlim(0, S.grid.L)
    phase_axes.set_ylim(-maxv, maxv)
    phase_axes.set_xlabel("x")
    phase_axes.set_ylabel("v_x")
    phase_axes.vlines(S.grid.x, -maxv, maxv)

    freq_plot, = freq_axes.plot([], [], "bo-", label="energy per mode")
    freq_axes.set_xlabel("k")
    freq_axes.set_ylabel("E")
    freq_axes.set_xlim(0, S.grid.NG/2)
    freq_axes.set_ylim(S.energy_per_mode.min(), S.energy_per_mode.max())

    plt.tight_layout()
    def init():
        iteration.set_text("Iteration: ")
        charge_plot.set_data([], [])
        field_plot.set_data([], [])
        freq_plot.set_data([], [])
        for species in S.all_species:
            phase_dots[species.name].set_data([], [])
            if lines:
                phase_lines[species.name].set_data([], [])
        if lines:
            return [charge_plot, field_plot, *phase_dots.values(),  iteration, *phase_lines.values()]
        else:
            return [charge_plot, field_plot, freq_plot, *phase_dots.values(),  iteration]

    def animate(i):
        charge_plot.set_data(S.grid.x, S.charge_density_history[i])
        field_plot.set_data(S.grid.x, S.electric_field_history[i])
        freq_plot.set_data(S.grid.k_plot, S.energy_per_mode[i])
        for species in S.all_species:
            phase_dots[species.name].set_data(S.position_history[species.name][i], S.velocity_history[species.name][i])
            if lines:
                phase_lines[species.name].set_data(S.position_history[species.name][:i + 1].T, S.velocity_history[species.name][:i + 1].T)
        iteration.set_text("Iteration: {}".format(i))

        if lines:
            return [charge_plot, field_plot, freq_plot, *phase_dots.values(),  iteration, *phase_lines.values()]
        else:
            return [charge_plot, field_plot, freq_plot, *phase_dots.values(),  iteration]

    animation_object = anim.FuncAnimation(fig, animate, interval=100, frames=int(S.NT), blit=True, init_func=init)
    if videofile_name:
        print("Saving animation to {}".format(videofile_name))
        animation_object.save(videofile_name, fps=15, writer='ffmpeg', extra_args=['-vcodec', 'libx264'])  # remove codecs to share video via IM
    plt.show()
    # return animation_object
