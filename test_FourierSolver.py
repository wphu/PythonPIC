from FourierSolver import *
import matplotlib.pyplot as plt
from helper_functions import l2_norm, l2_test
from Grid import Grid
DEBUG = False


def test_PoissonSolver(debug=DEBUG):
    def helper(NG, L):
        g = Grid(L, NG, epsilon_0=1)
        charge_density = (2 * np.pi / L)**2 * np.sin(2 * g.x * np.pi / L)
        field = -2 * np.pi / L * np.cos(2 * np.pi * g.x / L)
        potential = np.sin(2 * np.pi * g.x / L)
        g.charge_density = charge_density
        total_field_energy = g.solve_poisson()

        def plots():
            fig, axes = plt.subplots(3)
            ax0, ax1, ax2 = axes
            ax0.plot(g.x, charge_density)
            ax0.set_title("Charge density")
            ax1.set_title("Field")
            ax1.plot(g.x, g.electric_field, "r-", label="Fourier {:4.2f}".format(
                     l2_norm(field, g.electric_field)))
            ax1.plot(g.x, field, "g-", label="Analytic")
            ax2.set_title("Potential")
            ax2.plot(g.x, g.potential, "r-", label="Fourier {:4.2f}".format(
                     l2_norm(potential, g.potential)))
            ax2.plot(g.x, potential, "g-", label="Analytic")
            for ax in axes:
                ax.grid()
                ax.legend()
            plt.show()
            return "test_PoissonSolver failed! calc/theory field ratio at 0: {}".format(g.electric_field[0]/field[0])
        if debug:
            plots()
        field_correct = np.isclose(g.electric_field, field).all()
        potential_correct = np.isclose(g.potential, potential).all()
        assert field_correct and potential_correct, plots()
    helper(128, 1)
    helper(128, 2*np.pi)


# def test_PoissonSolver_complex(debug=DEBUG):
#     L = 1
#     N = 32 * 2**5
#     epsilon_0 = 1
#     x, dx = np.linspace(0, L, N, retstep=True, endpoint=False)
#     anal_potential = lambda x: np.sin(x * 2 * np.pi) + 0.5 * \
#         np.sin(x * 6 * np.pi) + 0.1 * np.sin(x * 20 * np.pi)
#     anal_field = lambda x: -(2 * np.pi * np.cos(x * 2 * np.pi) + 3 * np.pi *
#                    np.cos(x * 6 * np.pi) + 20 * np.pi * 0.1 * np.cos(x * 20 * np.pi))
#     charge_density_anal = lambda x: ((2 * np.pi)**2 * np.sin(x * 2 * np.pi) + 18 * np.pi**2 * np.sin(
#         x * 6 * np.pi) + (20 * np.pi)**2 * 0.1 * np.sin(x * 20 * np.pi)) * epsilon_0
#
#     NG = 32
#     g = Grid(L, NG, epsilon_0)
#     # indices_in_denser_grid = np.searchsorted(x, g.x)
#     g.charge_density = charge_density_anal(g.x)
#     energy_fourier = g.solve_poisson()
#     energy_direct = 0.5 * (g.electric_field**2).sum() * g.dx
#     print("dx", dx, "fourier", energy_fourier, "direct", energy_direct, energy_fourier / energy_direct)
#
#     def plots():
#         fig, xspace = plt.subplots()
#         xspace.set_title(
#             r"Solving the Poisson equation $\Delta \psi = \rho / \epsilon_0$ via Fourier transform")
#         xspace.plot(g.x, g.charge_density, "ro--", label=r"$\rho$")
#         xspace.plot(x, charge_density_anal(x), "r-", lw=6, alpha=0.5, label=r"$\rho_a$")
#         xspace.plot(g.x, g.potential, "go--", label=r"$V$")
#         xspace.plot(x, anal_potential(x), "g-", lw=6, alpha=0.5, label=r"$V_a$")
#         xspace.plot(g.x, g.electric_field, "bo--", alpha=0.5, label=r"$E$")
#         EplotAnal, = xspace.plot(x, anal_field(x), "b-", lw=6, alpha=0.5, label=r"$E_a$")
#         xspace.set_xlim(0, L)
#         xspace.set_xlabel("$x$")
#         xspace.grid()
#         xspace.legend(loc='best')
#
#         fig2, fspace = plt.subplots()
#         fspace.plot(g.k_plot, g.energy_per_mode, "bo--", label=r"electric energy $\rho_F V_F^\dagger$")
#         fspace.set_xlabel("k")
#         fspace.set_ylabel("mode energy")
#         fspace.set_title("Fourier space")
#         fspace.grid()
#         fspace.legend(loc='best')
#         plt.show()
#         return "test_PoissonSolver_complex failed!"
#
#     energy_correct = np.isclose(energy_fourier, energy_direct)
#     field_correct = np.isclose(g.electric_field, anal_field(g.x)).all()
#     potential_correct = np.isclose(g.potential, anal_potential(g.x)).all()
#     assert field_correct and potential_correct and energy_correct, plots()


def test_PoissonSolver_energy_sine(debug=DEBUG):
    L = 1
    N = 32 * 2**5
    epsilon_0 = 1
    x, dx = np.linspace(0, L, N, retstep=True, endpoint=False)
    anal_potential = np.sin(x * 2 * np.pi)
    anal_field = -(2 * np.pi * np.cos(x * 2 * np.pi))
    charge_density_anal = ((2 * np.pi)**2 * np.sin(x * 2 * np.pi))

    NG = 32
    g = Grid(L, NG, epsilon_0)
    indices_in_denser_grid = np.searchsorted(x, g.x)
    g.charge_density = charge_density_anal[indices_in_denser_grid]
    energy_fourier = g.solve_poisson()
    energy_direct = 0.5 * (g.charge_density * g.potential).sum() * g.dx
    print("dx", dx, "fourier", energy_fourier, "direct", energy_direct, energy_fourier / energy_direct)

    def plots():
        fig, xspace = plt.subplots()
        xspace.set_title(
            r"Solving the Poisson equation $\Delta \psi = \rho / \epsilon_0$ via Fourier transform")
        xspace.plot(g.x, g.charge_density, "ro--", label=r"$\rho$")
        xspace.plot(x, charge_density_anal, "r-", lw=6, alpha=0.5, label=r"$\rho_a$")
        xspace.plot(g.x, g.potential, "go--", label=r"$V$")
        xspace.plot(x, anal_potential, "g-", lw=6, alpha=0.5, label=r"$V_a$")
        xspace.plot(g.x, g.electric_field, "bo--", alpha=0.5, label=r"$E$")
        EplotAnal, = xspace.plot(x, anal_field, "b-", lw=6, alpha=0.5, label=r"$E_a$")
        xspace.set_xlim(0, L)
        xspace.set_xlabel("$x$")
        xspace.grid()
        xspace.legend(loc='best')

        fig2, fspace = plt.subplots()
        fspace.plot(g.k_plot, g.energy_per_mode, "bo--", label=r"electric energy $\rho_F V_F^\dagger$")
        fspace.set_xlabel("k")
        fspace.set_ylabel("mode energy")
        fspace.set_title("Fourier space")
        fspace.grid()
        fspace.legend(loc='best')
        plt.show()
        return "test_PoissonSolver_complex failed!"
    if debug:
        plots()
    print(g.electric_field - anal_field[indices_in_denser_grid])
    print(g.potential - anal_potential[indices_in_denser_grid])

    energy_correct = l2_test(energy_fourier, energy_direct)
    field_correct = l2_test(g.electric_field, anal_field[indices_in_denser_grid])
    potential_correct = l2_test(g.potential, anal_potential[indices_in_denser_grid])
    assert energy_correct, plots()
    assert potential_correct, plots()
    assert field_correct, plots()


def test_PoissonSolver_sheets(debug=DEBUG, test_charge_density=1):
    NG = 128
    L = 1
    epsilon_0 = 1

    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
    charge_density = np.zeros_like(x)
    region1 = (L * 1 / 8 < x) * (x < L * 2 / 8)
    region2 = (L * 5 / 8 < x) * (x < L * 6 / 8)
    charge_density[region1] = test_charge_density
    charge_density[region2] = -test_charge_density
    g = Grid(L, NG, epsilon_0)
    g.charge_density = charge_density
    g.solve_poisson()

    def plots():
        fig, axes = plt.subplots(3)
        ax0, ax1, ax2 = axes
        ax0.plot(x, charge_density)
        ax0.set_title("Charge density")
        ax1.set_title("Field")
        ax1.plot(x, g.electric_field, "r-")
        # ax1.plot(x, field, "g-", label="Analytic")
        ax2.set_title("Potential")
        ax2.plot(x, g.potential, "r-")
        # ax2.plot(x, potential, "g-", label="Analytic")
        for ax in axes:
            ax.grid()
            ax.legend()
        plt.show()
        return "test_PoissonSolver_sheets failed!"
    if debug:
        plots()

    polynomial_coefficients = np.polyfit(x[region1], g.electric_field[region1], 1)
    first_bump_right = np.isclose(
        polynomial_coefficients[0], test_charge_density, rtol=1e-2)
    polynomial_coefficients = np.polyfit(x[region2], g.electric_field[region2], 1)
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

    g = Grid(L, NG, epsilon_0=1)
    g.charge_density = a * g.x
    g.solve_poisson()
    potential = -a * g.x**3 / 6

    def plots():
        fig, axes = plt.subplots(3)
        ax0, ax1, ax2 = axes
        ax0.plot(g.x, g.charge_density)
        ax0.set_title("Charge density")
        ax1.set_title("Field")
        ax1.plot(g.x, g.electric_field, "r-")
        ax2.set_title("Potential")
        ax2.plot(g.x, g.potential, "r-")
        ax2.plot(g.x, potential, "g-", label="Analytic")
        for ax in axes:
            ax.grid()
            ax.legend()
        plt.show()
        return "test_PoissonSolver_ramp failed!"
    if debug:
        plots()

    polynomial_coefficients = np.polyfit(g.x, g.potential, 3)
    assert np.isclose(polynomial_coefficients[0], -a / 6,), plots()

if __name__ == "__main__":
    test_PoissonSolver_complex(True)
