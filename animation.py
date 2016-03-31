import matplotlib.pyplot as plt
import numpy as np
import Simulation
import matplotlib.animation as anim
S = Simulation.load_data("test.hdf5")

# fig, subplots = plt.subplots(3,2, squeeze=True, figsize=(20,20))
# (charge_axes, phase_axes), (field_axes, d3), (position_hist_axes, velocity_hist_axes) = subplots
fig, subplots = plt.subplots(3, squeeze=True, figsize=(10,5))
charge_axes, field_axes, phase_axes = subplots

phase_plot, = phase_axes.plot([], [], "b,")
phase_axes.set_xlim(0, S.L)
maxv =5* np.mean(np.abs(S.particle_velocities))
phase_axes.set_ylim(-maxv, maxv)
# fig.subplots_adjust(hspace=0)

charge_plot, = charge_axes.plot([],[])
# potential_plot, = charge_axes.plot(S.x, S.potential, "g-")

# charge_grid_scatter = charge_axes.scatter(S.x, np.zeros_like(S.x))

# position_histogram = position_hist_axes.hist(x_particles,NG, alpha=0.1)

# position_hist_axes.set_ylabel("$N$ at $x$")
# position_hist_axes.set_xlim(0,L)

field_plot, = field_axes.plot([],[])

# velocity_hist = velocity_hist_axes.hist(np.abs(v_particles),100)

# velocity_hist_axes.set_xlabel("$x$")
# velocity_hist_axes.set_xlabel("$v$")
# velocity_hist_axes.set_ylabel("$N$ at $v$")

# phase_axes_scatter = phase_axes.scatter([], [])

field_axes.set_ylabel(r"Field $E$")
field_axes.set_xlim(0,S.L)
charge_axes.set_xlim(0,S.L)
charge_axes.set_ylabel(r"Charge density $\rho$, potential $V$")
maxcharge = np.max(np.abs(S.charge_density))
charge_axes.set_ylim(-maxcharge, maxcharge)
maxfield = np.max(np.abs(S.electric_field))
field_axes.set_ylim(-maxfield, maxfield)
def init():
    charge_plot.set_data([], [])
    field_plot.set_data([], [])
    # phase_axes_scatter.set_array([], [])

    # phase_axes.set_xlim(0,S.L)
    field_axes.set_ylabel(r"Field $E$")
    field_axes.set_xlim(0,S.L)
    charge_axes.set_xlim(0,S.L)
    charge_axes.set_ylabel(r"Charge density $\rho$, potential $V$")
    return charge_plot, field_plot,  field_axes, charge_axes #phase_axes,
def animate(i):
    charge_plot.set_data(S.x, S.charge_density[i])
    phase_plot.set_data(S.particle_positions[i], S.particle_velocities[i])
    # position_histogram.set_data(S.x_particles, NG)
    field_plot.set_data(S.x, S.electric_field[i])
    # phase_axes_scatter.set_array(S.particle_positions[i], S.particle_velocities[i])
    return charge_plot, field_plot, phase_plot
animation = anim.FuncAnimation(fig, animate, interval=100, frames=S.NT,blit=True)
# animation.save("video.mp4", fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
