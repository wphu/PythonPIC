import Simulation
import numpy as np
import matplotlib.pyplot as plt

S = Simulation.load_data("data_analysis/TS5.hdf5")

for i in np.linspace(0, S.grid.NG, 10, endpoint=False, dtype=int):
    for data in [S.grid.charge_density_history[:,i],
                 S.grid.electric_field_history[:,i],
                 ]:
        time_fft = (np.fft.rfft(data).real)**2
        print(time_fft)
        fft_indices = np.fft.rfftfreq(S.NT, S.dt)
        plt.semilogy(fft_indices, time_fft)
    plt.title(i)
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
