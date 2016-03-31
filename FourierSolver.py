import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt

def PoissonSolver(rho, x, return_potential=False, epsilon_0=1):
    """Solves the Poisson equation on uniform periodic grids via Fourier Transform
    Assumes neutrality (0 fourier component of rho == 0)

    rho: array of charge density
    x: array of x positions
    return_potential: flag for returning potential, for plotting\purposes
    epsilon_0: the physical constant
    """

    NG = len(x)
    dx = x[1]-x[0]
    rho_F = fft.rfft(rho)
    rho_F[0] = 0
    k = fft.rfftfreq(NG, dx)
    potential_F = rho_F[:]
    potential_F[1:] /= k[1:]**2 * epsilon_0
    potential_F[0] = 0 #ignore
    potential = fft.irfft(potential_F)
    field = -np.gradient(potential)
    return field, potential
def PureFieldPoissonSolver(rho, x, epsilon_0 = 1):
    NG = len(x)
    dx = x[1]-x[0]
    rho_F = fft.fft(rho)
    k = fft.fftfreq(NG,dx)
    field_F = rho_F[:]
    field_F[1:] /= 1j*k[1:] * epsilon_0
    field = fft.irfft(field_F).real()
    return field
if __name__=="__main__":
    L=1
    N=1000
    epsilon_0 = 1
    x, dx = np.linspace(0, L, N, retstep=True, endpoint=False)
    charge_density = np.sin(x*2*np.pi)+0.5*np.sin(x*6*np.pi)+0.1*np.sin(x*20*np.pi)
    # charge_density = np.exp(-(x-L/2)**2/0.10)


    rho_F = fft.rfft(charge_density)
    k = fft.rfftfreq(N, dx)
    dk = k[1]-k[0]



    potential_F = np.empty_like(k)
    potential_F[0]=0
    potential_F[1:] = rho_F[1:]/k[1:]**2/epsilon_0

    field, potential = PoissonSolver(charge_density, x, return_potential=True)

    fig, (xspace, kspace) = plt.subplots(2,1)
    xspace.set_title(r"Solving the Poisson equation $\Delta \psi = \rho / \epsilon_0$ via Fourier transform")

    kspace.set_title(r"In Fourier space, $\Delta = k^{-2}$")
    rhoFplot = kspace.bar(k, rho_F, dk, linewidth=0, color=[1,0,0], alpha=0.5, label=r"$\rho_F$")
    VFplot = kspace.bar(k, potential_F, dk, linewidth=0, color=[0,1,0], alpha=0.5, label=r"$V_F$")
    kspace.legend((rhoFplot, VFplot), (r"$\rho_F$",r"$V_F$"))
    kspace.set_xlim(0,20)
    kspace.set_xlabel(r"$k$")
    kspace.grid()


    rhoplot, = xspace.plot(x, charge_density, "r-", label=r"\rho")
    Vplot, = xspace.plot(x, potential, "g-", label=r"$V$")
    Eplot, = xspace.plot(x, field/np.max(field), "b-", label=r"$E$ (scaled)")
    EplotNoScale, = xspace.plot(x, field, "b--", label=r"$E$ (not scaled)")
    xspace.set_xlim(0,L)
    xspace.set_xlabel("$x$")
    xspace.grid()
    xspace.legend((rhoplot, Vplot, Eplot, EplotNoScale), (r"$\rho$", r"$V$", r"$E$ (scaled)", "$E$ (not scaled)"))
    plt.savefig("FourierSolver.pdf")
    plt.show()
