from FourierSolver import *
import matplotlib.pyplot as plt
from helper_functions import l2_norm

DEBUG = False


def test_PoissonSolver(debug=DEBUG):
    NG = 128
    L = 1

    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)

    charge_density = (2 * np.pi)**2 * np.sin(2 * x * np.pi)
    field = -2 * np.pi * np.cos(2 * np.pi * x)
    potential = np.sin(2 * np.pi * x)

    FSfield, FSpotential, FSenergy = PoissonSolver(charge_density, x)

    def plots():
        fig, axes = plt.subplots(3)
        ax0, ax1, ax2 = axes
        ax0.plot(x, charge_density)
        ax0.set_title("Charge density")
        ax1.set_title("Field")
        ax1.plot(
            x,
            FSfield,
            "r-",
            label="Fourier {:4.2f}".format(
                l2_norm(
                    field,
                    FSfield)))
        ax1.plot(x, field, "g-", label="Anal")
        ax2.set_title("Potential")
        ax2.plot(
            x,
            FSpotential,
            "r-",
            label="Fourier {:4.2f}".format(
                l2_norm(
                    potential,
                    FSpotential)))
        ax2.plot(x, potential, "g-", label="Anal")
        for ax in axes:
            ax.grid()
            ax.legend()
        plt.show()
        return "test_PoissonSolver failed!"

    field_correct = np.isclose(FSfield, field).all()
    potential_correct = np.isclose(FSpotential, potential).all()
    assert field_correct and potential_correct, plots()


def test_PoissonSolver_complex(debug=DEBUG):
    L = 1
    N = 32 * 2**5
    epsilon_0 = 1
    x, dx = np.linspace(0, L, N, retstep=True, endpoint=False)
    anal_potential = np.sin(x * 2 * np.pi) + 0.5 * \
        np.sin(x * 6 * np.pi) + 0.1 * np.sin(x * 20 * np.pi)
    anal_field = -(2 * np.pi * np.cos(x * 2 * np.pi) + 3 * np.pi *
                   np.cos(x * 6 * np.pi) + 20 * np.pi * 0.1 * np.cos(x * 20 * np.pi))
    charge_density_anal = ((2 * np.pi)**2 * np.sin(x * 2 * np.pi) + 18 * np.pi**2 * np.sin(
        x * 6 * np.pi) + (20 * np.pi)**2 * 0.1 * np.sin(x * 20 * np.pi)) * epsilon_0

    NG = 32
    x_grid, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
    indices_in_denser_grid = np.searchsorted(x, x_grid)
    charge_density = charge_density_anal[indices_in_denser_grid]
    field, potential, energy_presum, k = PoissonSolver(charge_density, x_grid, debug=True)
    energy_fourier = energy_presum.sum()
    energy_direct = 0.5 * (field**2).sum()
    print("dx", dx, "fourier", energy_fourier, "direct", energy_direct, energy_fourier / energy_direct)

    def plots():
        fig, xspace = plt.subplots()
        xspace.set_title(
            r"Solving the Poisson equation $\Delta \psi = \rho / \epsilon_0$ via Fourier transform")
        rhoplot, = xspace.plot(x_grid, charge_density, "ro--", label=r"$\rho$")
        rhoplotAnal, = xspace.plot(x, charge_density_anal, "r-", lw=6, alpha=0.5, label=r"$\rho_a$")
        Vplot, = xspace.plot(x_grid, potential, "go--", label=r"$V$")
        VplotAnal, = xspace.plot(x, anal_potential, "g-", lw=6, alpha=0.5, label=r"$V_a$")
        Eplot, = xspace.plot(x_grid, field, "bo--", alpha=0.5, label=r"$E$")
        EplotAnal, = xspace.plot(x, anal_field, "b-", lw=6, alpha=0.5, label=r"$E_a$")
        xspace.set_xlim(0, L)
        xspace.set_xlabel("$x$")
        xspace.grid()
        xspace.legend(loc='best')

        fig2, fspace = plt.subplots()
        fspace.plot(k, energy_presum, "bo--", label=r"electric energy $\rho_F V_F^\dagger$")
        # fspace.plot(k, energy_via_field, "go--", label="energy via field?")
        fspace.set_xlabel("k")
        fspace.set_ylabel("mode energy")
        fspace.set_title("Fourier space")
        fspace.grid()
        fspace.legend(loc='best')
        plt.show()
        return "test_PoissonSolver_complex failed!"
    print(field - anal_field[indices_in_denser_grid])
    print(potential - anal_potential[indices_in_denser_grid])

    energy_correct = np.isclose(energy_fourier, energy_direct)
    field_correct = np.isclose(field, anal_field[indices_in_denser_grid]).all()
    potential_correct = np.isclose(potential, anal_potential[indices_in_denser_grid]).all()
    assert field_correct and potential_correct and energy_correct, plots()
    # assert False, plots()


def test_PoissonSolver_sheets(debug=DEBUG, test_charge_density=1):
    NG = 128
    L = 1

    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
    charge_density = np.zeros_like(x)
    region1 = (L * 1 / 8 < x) * (x < L * 2 / 8)
    region2 = (L * 5 / 8 < x) * (x < L * 6 / 8)
    charge_density[region1] = test_charge_density
    charge_density[region2] = -test_charge_density

    FSfield, FSpotential, FSenergy = PoissonSolver(charge_density, x)

    def plots():
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
        return "test_PoissonSolver_sheets failed!"

    polynomial_coefficients = np.polyfit(x[region1], FSfield[region1], 1)
    first_bump_right = np.isclose(
        polynomial_coefficients[0],
        test_charge_density,
        rtol=1e-2)
    polynomial_coefficients = np.polyfit(x[region2], FSfield[region2], 1)
    second_bump_right = np.isclose(
        polynomial_coefficients[0], -test_charge_density, rtol=1e-2)
    assert first_bump_right and second_bump_right, plots()


def test_PoissonSolver_ramp(debug=DEBUG):
    """ For a charge density rho = Ax + B
    d2phi/dx2 = -rho/epsilon_0
    set epsilon_0 to 1
    d2phi/dx2 = Ax
    phi must be of form
    phi = -Ax^3/6 + Bx^2 + Cx + D"""

    NG = 128
    L = 1

    a = 1

    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
    charge_density = a * x

    FSfield, FSpotential, FSenergy = PoissonSolver(charge_density, x)
    potential = -a * x**3 / 6

    def plots():
        fig, axes = plt.subplots(3)
        ax0, ax1, ax2 = axes
        ax0.plot(x, charge_density)
        ax0.set_title("Charge density")
        ax1.set_title("Field")
        ax1.plot(x, FSfield, "r-")
        ax2.set_title("Potential")
        ax2.plot(x, FSpotential, "r-")
        ax2.plot(x, potential, "g-", label="Anal")
        for ax in axes:
            ax.grid()
            ax.legend()
        plt.show()
        return "test_PoissonSolver_ramp failed!"

    polynomial_coefficients = np.polyfit(x, FSpotential, 3)
    assert np.isclose(polynomial_coefficients[0], -a / 6,), plots()

if __name__ == "__main__":
    test_PoissonSolver_complex(True)
