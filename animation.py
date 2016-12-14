import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

colors = "brgyk"

def animation(S, videofile_name, lines=False, alpha=1):
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
    mincharge = np.min(S.grid.charge_density_history)
    maxcharge = np.max(S.grid.charge_density_history)
    charge_axes.set_ylim(mincharge, maxcharge)
    charge_axes.set_xlim(0, S.grid.L)
    charge_axes.set_ylabel(r"Charge density $\rho$")
    charge_axes.vlines(S.grid.x, mincharge, maxcharge)

    field_axes.vlines(S.grid.x, -1, 1)
    field_plot, = field_axes.plot([], [])
    field_axes.set_ylabel(r"Field $E$")
    field_axes.set_xlim(0, S.grid.L)
    maxfield = np.max(np.abs(S.grid.electric_field_history))
    field_axes.set_ylim(-maxfield, maxfield)
    field_axes.set_ylabel(r"Field $E$")
    field_axes.set_xlim(0, S.grid.L)

    phase_dots = {}
    if lines:
        phase_lines = {}
    for i, species in enumerate(S.all_species):
        phase_dots[species.name], = phase_axes.plot([], [], colors[i]+".", alpha=alpha)
        if lines:
            phase_lines[species.name], = phase_axes.plot([], [], colors[i]+"-", alpha=alpha/2, lw=0.7)
    maxv = max([5 * np.mean(np.abs(species.velocity_history)) for species in S.all_species])
    phase_axes.set_xlim(0, S.grid.L)
    phase_axes.set_ylim(-maxv, maxv)
    phase_axes.set_xlabel("x")
    phase_axes.set_ylabel("v_x")
    phase_axes.vlines(S.grid.x, -maxv, maxv)

    freq_plot, = freq_axes.plot([], [], "bo-", label="energy per mode")
    freq_axes.set_xlabel("k")
    freq_axes.set_ylabel("E")
    freq_axes.set_xlim(0, S.grid.NG/2)
    freq_axes.set_ylim(S.grid.energy_per_mode_history.min(), S.grid.energy_per_mode_history.max())

    plt.tight_layout()
    def init():
        iteration.set_text("Iteration: ")
        charge_plot.set_data(S.grid.x, np.zeros_like(S.grid.x))
        field_plot.set_data(S.grid.x, np.zeros_like(S.grid.x))
        freq_plot.set_data(S.grid.k_plot, np.zeros_like(S.grid.k_plot))
        for species in S.all_species:
            phase_dots[species.name].set_data([], [])
            if lines:
                phase_lines[species.name].set_data([], [])
        if lines:
            return [charge_plot, field_plot, *phase_dots.values(),  iteration, *phase_lines.values()]
        else:
            return [charge_plot, field_plot, freq_plot, *phase_dots.values(),  iteration]

    def animate(i):
        # import ipdb; ipdb.set_trace()
        charge_plot.set_ydata(S.grid.charge_density_history[i])
        field_plot.set_ydata(S.grid.electric_field_history[i])
        freq_plot.set_ydata(S.grid.energy_per_mode_history[i])
        for species in S.all_species:
            phase_dots[species.name].set_data(species.position_history[i,:], species.velocity_history[i,:,0])
            if lines:
                phase_lines[species.name].set_data(species.position_history[:i + 1, :].T, species.velocity_history[:i + 1, :, 0].T)
        iteration.set_text("Iteration: {}".format(i))

        if lines:
            return [charge_plot, field_plot, freq_plot, *phase_dots.values(),  iteration, *phase_lines.values()]
        else:
            return [charge_plot, field_plot, freq_plot, *phase_dots.values(),  iteration]

    animation_object = anim.FuncAnimation(fig, animate, interval=100, frames=int(S.NT), blit=True, init_func=init)
    if videofile_name:
        print("Saving animation to {}".format(videofile_name))
        animation_object.save(videofile_name, fps=15, writer='ffmpeg', extra_args=['-vcodec', 'libx264'])
    plt.show()

if __name__=="__main__":
    import Simulation
    S = Simulation.load_data("data_analysis/TS3.hdf5")
    animation(S, None, False)
    if show:
        plt.show()
