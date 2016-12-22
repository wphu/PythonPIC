import numpy as np
import matplotlib.pyplot as plt
import Simulation


def get_fft(filename):
    S = Simulation.load_data(filename)
    omega = np.arange(S.NT)
    k = np.arange(S.grid.NG)
    charge_density = S.grid.charge_density_history
    K, OMEGA = np.meshgrid(k, omega)
    fft = 20*np.log(np.abs(np.fft.fftn(charge_density)))
    return K, OMEGA, fft

def spectro_1d(filename, i):
    K, OMEGA, fft = get_fft(filename)
    omega = OMEGA[:,i]
    plt.stem(omega,fft[:,i])
    plt.show()


def spectro_2d(filename):
    K, OMEGA, fft = get_fft(filename)
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(fft[:,1:], aspect='auto', interpolation='none', origin='lower', cmap='viridis'),
    # plt.colorbar(IM)
    plt.title(filename)
    plt.xlabel("k")
    plt.ylabel("omega")
    ax2.contourf(K[:,1:], OMEGA[:,1:], fft[:,1:], aspect='auto', origin='lower', cmap='viridis')
    plt.show()
if __name__ == '__main__':
    for i in range(1,3):
        filename = "data_analysis/TS{}.hdf5".format(i)
        spectro_2d(filename)
