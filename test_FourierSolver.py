from FourierSolver import *

DEBUG = True

def test_PoissonSolver(debug=DEBUG):
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


def test_PoissonSolver_complex(debug=DEBUG):
    L=1
    N=1000
    epsilon_0 = 1
    x, dx = np.linspace(0, L, N, retstep=True, endpoint=False)
    anal_potential = np.sin(x*2*np.pi)+0.5*np.sin(x*6*np.pi)+0.1*np.sin(x*20*np.pi)
    anal_field = -(2*np.pi*np.cos(x*2*np.pi)+3*np.pi*np.cos(x*6*np.pi)+20*np.pi*0.1*np.cos(x*20*np.pi))
    charge_density = ((2*np.pi)**2*np.sin(x*2*np.pi)+18*np.pi**2*np.sin(x*6*np.pi)+(20*np.pi)**2*0.1*np.sin(x*20*np.pi))*epsilon_0

    field, potential, energy = PoissonSolver(charge_density, x)

    if debug:
        fig, xspace = plt.subplots()
        xspace.set_title(r"Solving the Poisson equation $\Delta \psi = \rho / \epsilon_0$ via Fourier transform")
        rhoplot, = xspace.plot(x, charge_density, "r-", label=r"\rho")
        Vplot, = xspace.plot(x, potential, "g-", label=r"$V$")
        Eplot, = xspace.plot(x, field/np.max(np.abs(field)), "b-", label=r"$E$ (scaled)")
        EplotNoScale, = xspace.plot(x, field, "b--", label=r"$E$ (not scaled)")
        xspace.set_xlim(0,L)
        xspace.set_xlabel("$x$")
        xspace.grid()
        xspace.legend((rhoplot, Vplot, Eplot, EplotNoScale), (r"$\rho$", r"$V$", r"$E$ (scaled)", "$E$ (not scaled)"))
        plt.show()
    print(field-anal_field)
    print(potential-anal_potential)
    assert np.logical_and(np.isclose(field, anal_field).all(), np.isclose(potential, anal_potential).all())

def test_PoissonSolver_sheets(debug=DEBUG, test_charge_density=1):
    from diagnostics import L2norm
    NG = 128
    L = 1

    x, dx = np.linspace(-L/2,L/2,NG, retstep=True,endpoint=False)
    charge_density = np.zeros_like(x)
    region1 = (-L*1/4 < x) * (x < -L*1/8)
    region2 = (L*1/8 < x) * (x < L*1/4)
    charge_density[region1] = test_charge_density
    charge_density[region2] = -test_charge_density

    FSfield, FSpotential, FSenergy = PoissonSolver(charge_density, x)

    if debug:
        fig, axes = plt.subplots(3)
        ax0, ax1, ax2 = axes
        ax0.plot(x, charge_density)
        ax0.set_title("Charge density")
        ax1.set_title("Field")
        ax1.plot(x, FSfield, "r-")
        # ax1.plot(x, field, "g-", label="Anal")
        ax2.set_title("Potential")
        ax2.plot(x, FSpotential, "r-")
        # ax2.plot(x, potential, "g-", label="Anal")
        for ax in axes:
            ax.grid()
            ax.legend()
        plt.show()

    polynomial_coefficients = np.polyfit(x[region1], FSfield[region1], 1)
    # print(polynomial_coefficients[0], -test_charge_density)
    assert np.isclose(polynomial_coefficients[0], test_charge_density, rtol=1e-2)

    polynomial_coefficients = np.polyfit(x[region2], FSfield[region2], 1)
    # print(polynomial_coefficients[0])
    assert np.isclose(polynomial_coefficients[0], -test_charge_density, rtol=1e-2)

def test_PoissonSolver_ramp(debug=DEBUG):
    """ For a charge density rho = Ax + B
    d2phi/dx2 = -rho/epsilon_0
    set epsilon_0 to 1
	d2phi/dx2 = Ax
	phi must be of form
	phi = -Ax^3/6 + Bx^2 + Cx + D"""


    from diagnostics import L2norm
    NG = 128
    L = 1

    a = 1

    x, dx = np.linspace(-L/2,L/2,NG, retstep=True,endpoint=False)
    charge_density = a*x

    FSfield, FSpotential, FSenergy = PoissonSolver(charge_density, x)
    potential = -a*x**3/6
    if debug:
        fig, axes = plt.subplots(3)
        ax0, ax1, ax2 = axes
        ax0.plot(x, charge_density)
        ax0.set_title("Charge density")
        ax1.set_title("Field")
        ax1.plot(x, FSfield, "r-")
        # ax1.plot(x, field, "g-", label="Anal")
        ax2.set_title("Potential")
        ax2.plot(x, FSpotential, "r-")
        ax2.plot(x, potential, "g-", label="Anal")
        for ax in axes:
            ax.grid()
            ax.legend()
        plt.show()

    polynomial_coefficients = np.polyfit(x, FSpotential, 3)
    # print(polynomial_coefficients[0], -a/6)
    assert np.isclose(polynomial_coefficients[0], -a/6,)

if __name__=="__main__":
	test_PoissonSolver_sheets()
