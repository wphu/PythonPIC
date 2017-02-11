import Simulation
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

S = Simulation.load_data("data_analysis/CO1.hdf5")


charge_fft = (np.abs(np.fft.rfft(S.grid.charge_density_history, axis=0)))**2
field_fft = (np.abs(np.fft.rfft(S.grid.electric_field_history, axis=0)))**2

fft_indices = np.fft.rfftfreq(S.NT)
charge_mean_fft = charge_fft.mean(axis=1)
field_mean_fft = field_fft.mean(axis=1)

plt.semilogy(fft_indices, field_fft)
plt.semilogy(fft_indices, field_mean_fft, lw=5)
plt.grid()
plt.show()

# k = np.fft.rfftfreq(sim.NT/2+1, sim.dt)
# k = np.arange(S.NT/2+1)
# omega = np.arange(S.grid.NG/2)
# omega = np.fft.rfftfreq(int(sim.grid.NG)-1, sim.grid.dx)
# K, OMEGA = np.meshgrid(omega, k)

# plt.contourf(K, OMEGA, time_fft, contours=20)
# plt.imshow(time_fft, aspect='equal', interpolation=None)
# plt.ion()
# plt.colorbar()
# plt.show()

# plt.bar(left_edges, time_fft[0], log=True)
# plt.grid()
# plt.xlabel("k")
# plt.ylabel("$\omega$")
# plt.show()
