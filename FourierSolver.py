import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt

def PoissonSolver(rho, x, epsilon_0 = 1):
    #TODO: $G(k) = delta_x sum_{j=0}^{NG-1} G(X_j) exp(-ikX_j)$
    # so for k = 0, this reduces to
    # $delta_x$ sum_{gridpoints} G(X_j)$
    # this happens to be the integral of the thing
    # does this let us regain potential(k=0)?

    #TODO: Aliasing: keep abs(delta_x k) < pi
    NG = len(x)
    dx = x[1]-x[0]
    rho_F = fft.fft(rho)
    k = fft.fftfreq(NG,dx)
    field_F = np.zeros_like(rho_F)
    field_F[1:] = rho_F[1:]/(np.pi*2j*k[1:] * epsilon_0)
    potential_F = np.zeros_like(rho_F)
    potential_F[1:] = field_F[1:]/(-2j*np.pi*k[1:] * epsilon_0)
    field = fft.ifft(field_F).real
    potential = fft.ifft(potential_F).real

    energy = (0.5*np.sum(rho_F*potential_F.conjugate())).real

    return field, potential, energy

def PoissonSolver_test(debug=False):
    from diagnostics import L2norm
    NG = 128
    L = 1

    x, dx = np.linspace(-L/2,L/2,NG, retstep=True,endpoint=False)
    charge_density = np.zeros_like(x)

    charge_density = (2*np.pi)**2*np.sin(2*x*np.pi)
    field = -2*np.pi*np.cos(2*np.pi*x)
    potential = np.sin(2*np.pi*x)

    FSfield, FSpotential, FSenergy = PoissonSolver(charge_density, x)

    if debug:
        fig, axes = plt.subplots(3)
        ax0, ax1, ax2 = axes
        ax0.plot(x, charge_density)
        ax0.set_title("Charge density")
        ax1.set_title("Field")
        ax1.plot(x, FSfield, "r-", label="Fourier {:4.2f}".format(L2norm(field, FSfield)))
        ax1.plot(x, field, "g-", label="Anal")
        ax2.set_title("Potential")
        ax2.plot(x, FSpotential, "r-", label="Fourier {:4.2f}".format(L2norm(potential, FSpotential)))
        ax2.plot(x, potential, "g-", label="Anal")
        for ax in axes:
            ax.grid()
            ax.legend()
        plt.show()
    assert np.logical_and(np.isclose(FSfield, field).all(), np.isclose(FSpotential, potential).all())


def PoissonSolver_complex_test(debug=False):
    L=1
    N=1000
    epsilon_0 = 1
    x, dx = np.linspace(0, L, N, retstep=True, endpoint=False)
    anal_potential = np.sin(x*2*np.pi)+0.5*np.sin(x*6*np.pi)+0.1*np.sin(x*20*np.pi)
    anal_field = -(2*np.pi*np.cos(x*2*np.pi)+3*np.pi*np.cos(x*6*np.pi)+20*np.pi*0.1*np.cos(x*20*np.pi))
    charge_density = ((2*np.pi)**2*np.sin(x*2*np.pi)+18*np.pi**2*np.sin(x*6*np.pi)+(20*np.pi)**2*0.1*np.sin(x*20*np.pi))*epsilon_0

    field, potential = PoissonSolver(charge_density, x)

    if debug:
        fig, xspace = plt.subplots()
        xspace.set_title(r"Solving the Poisson equation $\Delta \psi = \rho / \epsilon_0$ via Fourier transform")
        rhoplot, = xspace.plot(x, charge_density, "r-", label=r"\rho")
        Vplot, = xspace.plot(x, potential, "g-", label=r"$V$")
        Eplot, = xspace.plot(x, field/np.max(field), "b-", label=r"$E$ (scaled)")
        EplotNoScale, = xspace.plot(x, field, "b--", label=r"$E$ (not scaled)")
        xspace.set_xlim(0,L)
        xspace.set_xlabel("$x$")
        xspace.grid()
        xspace.legend((rhoplot, Vplot, Eplot, EplotNoScale), (r"$\rho$", r"$V$", r"$E$ (scaled)", "$E$ (not scaled)"))
        plt.show()
    print(field-anal_field)
    print(potential-anal_potential)
    assert np.logical_and(np.isclose(field, anal_field).all(), np.isclose(potential, anal_potential).all())
