import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.fftpack as fft
import parameters


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="hdf5 file name for storing data")
args = parser.parse_args()
if(args.filename[-5:] != ".hdf5"):
    args.filename = args.filename + ".hdf5"

with h5py.File(args.filename) as f:
    NT = int(f.attrs['NT'])
    dt = f.attrs['T']/NT
    fourier_frequencies = fft.rfftfreq(NT, dt)
    freqstep = fourier_frequencies[1]-fourier_frequencies[0]
    print("NT: {}, dt: {}".format(NT, dt))
    fig, (timed, freqd) = plt.subplots(2)
    freqd.plot(fourier_frequencies,
        np.ones_like(fourier_frequencies)*parameters.plasma_frequency)
    for i in (0, 8, 16, 30):
        rho = charge_density = f['Charge density'][:,i]
        rho_F = fft.rfft(rho)
        rho_F[0]=0
        print("Discrepancy: {}".format(rho_F[0]/parameters.plasma_frequency))
        timed.plot(np.arange(0,NT*dt,dt),charge_density, label=i)
        freqd.bar(fourier_frequencies, rho_F, width=freqstep, label=i)
    timed.legend()
    freqd.legend()
    plt.show()
