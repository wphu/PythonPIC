import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def dispersion_relation(t, x, z, plot_spectro=False):
    # omega od k
    # dla każdego k
    # przejrzeć wszystkie omegi na osi 0
    # wybrać taką która ma maksymalną składową
    # plotnąć ją dla tego k
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    phase_resolution = np.pi * 2
    k = (np.fft.rfftfreq(x.size, dx)) * 2 * phase_resolution
    omega_vector = (np.fft.rfftfreq(t.size, dt)) * phase_resolution

    space_time_fft = np.fft.rfft(np.fft.rfft(z, axis=1), axis=0)
    plottable_space_time_fft = np.log(np.abs(space_time_fft).real)


    """
    signals are:
    in rows numbered by k_indices
    IF
    """
    noise_indices = plottable_space_time_fft < 0
    analysis_space_time_fft = plottable_space_time_fft.copy()
    analysis_space_time_fft[noise_indices] = 0
    maximal_omega_index = np.argmax(analysis_space_time_fft, axis=0)
    maximal_omega = omega_vector[maximal_omega_index]

    if plot_spectro:
        OMEGA, K = np.meshgrid(omega_vector, k, indexing='ij')
        plt.contourf(OMEGA, K, plottable_space_time_fft, 500)
        plt.colorbar()
        plt.plot(maximal_omega, k, "ro-")
        plt.xlabel("omega")
        plt.ylabel("k")
        plt.show()

    # import ipdb; ipdb.set_trace()
    plt.plot(k, maximal_omega)
    plt.xlabel("k")
    plt.ylabel("omega")
    plt.show()
    return k, maximal_omega

if __name__ == '__main__':
    tmax = 1
    xmax = 12
    t = np.linspace(0,tmax,128, endpoint=False)
    dt = t[1] - t[0]
    x = np.linspace(0,xmax,128, endpoint=False)
    dx = x[1] - x[0]

    T, X = np.meshgrid(t, x, indexing='ij')
    wavevector = 6 * np.pi
    omega = 10 * np.pi
    z = 5*np.cos(wavevector*X-omega*T) +\
        np.sin(wavevector*X + 40*np.pi * T) +\
        10*np.sin(8*np.pi * X - omega*T)

    dispersion_relation(t,x,z,plot_spectro=True)
