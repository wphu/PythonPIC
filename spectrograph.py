import numpy as np
import matplotlib.pyplot as plt
import Simulation

def dispersion_relation(t, x, z, plot_spectro=False, plot_dispersion=False):
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
        def plot_spectrograph(omega_vector, k, plottable_space_time_fft):
            OMEGA, K = np.meshgrid(omega_vector, k, indexing='ij')
            plt.contourf(OMEGA, K, plottable_space_time_fft, 500)
            plt.colorbar()
            plt.plot(maximal_omega, k, "ro-")
            plt.xlabel("omega")
            plt.ylabel("k")
            plt.show()
        plot_spectrograph(omega_vector, k, plottable_space_time_fft)

    if plot_dispersion:
        def plot_dispersion_relation(k, maximal_omega):
            plt.plot(k, maximal_omega)
            plt.xlabel("k")
            plt.ylabel("omega")
            plt.show()
        plot_dispersion_relation(k, maximal_omega)
    return k, maximal_omega

def test_spectrograph():
    tmax = 1
    xmax = 12
    t = np.linspace(0,tmax,128, endpoint=False)
    x = np.linspace(0,xmax,128, endpoint=False)

    T, X = np.meshgrid(t, x, indexing='ij')
    wavevector = 6 * np.pi
    omega = 10 * np.pi
    z = 5*np.cos(wavevector*X-omega*T) +\
        10*np.sin(8*np.pi * X - omega*T)

    result_k, result_omega = dispersion_relation(t,x,z,plot_spectro=True, plot_dispersion=True)
    assert (np.logical_or(np.isclose(result_omega, omega, rtol=1e-3),
            np.isclose(result_omega, 0))).all(), (result_omega, omega)

def spectral_analysis(filename):
    S = Simulation.load_data(filename)
    t = np.arange(S.NT+1)*S.dt
    dispersion_relation(t, S.grid.x, S.grid.charge_density_history, plot_spectro=True, plot_dispersion=True)
    dispersion_relation(t, S.grid.x, S.grid.electric_field_history, plot_spectro=True, plot_dispersion=True)

if __name__ == '__main__':
    for i in range(1,11):
        filename = "data_analysis/TS{}.hdf5".format(i)
        spectral_analysis(filename)
