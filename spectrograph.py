import numpy as np
import matplotlib.pyplot as plt
import Simulation

for i in range(1,11):
    filename = "data_analysis/TS{}.hdf5".format(i)
    S = Simulation.load_data(filename)
    omega = np.arange(S.NT)
    k = np.arange(S.grid.NG)
    charge_density = S.grid.charge_density_history
    K, OMEGA = np.meshgrid(k, omega)
    fft = np.log(np.abs(np.fft.fftn(charge_density))**2)
    plt.imshow(fft[:,1:], aspect='auto', origin='lower', vmin=-30, vmax=30, cmap='viridis')
    # , vmin=-100, vmax=30,
    # plt.contourf(K, OMEGA, fft, 100)
    plt.colorbar()
    plt.title(filename)
    plt.xlabel("k")
    plt.ylabel("omega")
    plt.savefig(filename.replace(".hdf5","_spectro.png"))
    plt.clf()
