import Simulation
import numpy as np
import matplotlib.pyplot as plt

sim = Simulation.load_data("data_analysis/1.hdf5")
energy_per_mode = sim.energy_per_mode
time_fft = np.fft.rfft(energy_per_mode, axis=0).real
# plt.semilogy(time_fft.T)
# k = np.fft.rfftfreq(sim.NT/2+1, sim.dt)
k = np.arange(sim.NT/2+1)
omega = np.arange(sim.grid.NG/2)
# omega = np.fft.rfftfreq(int(sim.grid.NG)-1, sim.grid.dx)
K, OMEGA = np.meshgrid(omega, k)

# plt.contourf(K, OMEGA, time_fft, contours=20)
plt.imshow(time_fft, aspect='equal', interpolation=None)
# plt.ion()
plt.colorbar()
plt.show()

# plt.bar(left_edges, time_fft[0], log=True)
# plt.grid()
# plt.xlabel("k")
# plt.ylabel("$\omega$")
# plt.show()
