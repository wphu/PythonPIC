import matplotlib.pyplot as plt
import numpy as np
import Database
import matplotlib.animation as anim
S = Database.load_data("test.hdf5")


fig, subplots = plt.subplots(3,2, squeeze=True, figsize=(20,20))
(charge_axes, phase_axes), (field_axes, d3), (position_hist_axes, velocity_hist_axes) = subplots
fig.subplots_adjust(hspace=0)

charge_plot, = charge_axes.plot(S.x,S.charge_density, label="charge density")
# potential_plot, = charge_axes.plot(S.x, S.potential, "g-")

charge_grid_scatter, = charge_axes.scatter(S.x, np.zeros_like(S.x))

# position_histogram = position_hist_axes.hist(x_particles,NG, alpha=0.1)

# position_hist_axes.set_ylabel("$N$ at $x$")
# position_hist_axes.set_xlim(0,L)

field_plot, = field_axes.plot(S.xelectric_field, label="electric field")

# velocity_hist = velocity_hist_axes.hist(np.abs(v_particles),100)

# velocity_hist_axes.set_xlabel("$x$")
# velocity_hist_axes.set_xlabel("$v$")
# velocity_hist_axes.set_ylabel("$N$ at $v$")

phase_axes_scatter, = phase_axes.scatter([], [])

def init(i):
    charge_plot.set_data([], [])
    field_plot.set_data([], [])
    phase_axes_scatter.set_data([], [])

    phase_axes.set_xlim(0,S.L)
    field_axes.set_ylabel(r"Field $E$")
    field_axes.set_xlim(0,S.L)
    charge_axes.set_xlim(0,S.L)
    charge_axes.set_ylabel(r"Charge density $\rho$, potential $V$")
def animate(i):
    charge_plot.set_data(S.x, S.charge_density[i])
    # position_histogram.set_data(S.x_particles, NG)
    field_plot.set_data(S.x, S.electric_field[i])
    phase_axes_scatter.set_data(S.particle_positions[i], S.particle_velocities[i])
animation = anim.FuncAnimation(fig, animate, frames=S.NT, init_func=init, blit=True)
plt.show()
