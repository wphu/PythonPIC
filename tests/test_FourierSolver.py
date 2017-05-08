# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pytest

from Grid import Grid
from helper_functions import l2_norm, l2_test

DEBUG = False


@pytest.mark.parametrize(["NG", "L"], [
    (128, 1),
    (128, 2 * np.pi)
    ])
def test_PoissonSolver(NG, L, debug=DEBUG):
    g = Grid(L, NG)
    charge_density = (2 * np.pi / L) ** 2 * np.sin(2 * g.x * np.pi / L)
    field = np.zeros((NG + 2, 3))
    field[1:-1, 0] = -2 * np.pi / L * np.cos(2 * np.pi * g.x / L)
    g.charge_density[1:-1] = charge_density
    g.solve()

    def plots():
        fig, axes = plt.subplots(2)
        ax0, ax1 = axes
        ax0.plot(g.x, charge_density)
        ax0.set_title("Charge density")
        ax1.set_title("Field")
        ax1.plot(g.x, g.electric_field[1:-1], "r-", label="Fourier {:4.2f}".format(
            l2_norm(field, g.electric_field[1:-1])))
        ax1.plot(g.x, field, "g-", label="Analytic")
        for ax in axes:
            ax.grid()
            ax.legend()
        plt.show()
        return "test_PoissonSolver failed! calc/theory field ratio at 0: {}".format(g.electric_field[1] / field[0])

    if debug:
        plots()
    field_correct = np.isclose(g.electric_field, field).all()
    assert field_correct, plots()


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
#     energy_fourier = g.solve_fourier()
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

@pytest.mark.parametrize(["NG", "L"], [
    (32, 1),
    ])
def test_PoissonSolver_energy_sine(NG, L, debug=DEBUG):
    resolution_increase = NG
    N = NG * resolution_increase
    epsilon_0 = 1
    x, dx = np.linspace(0, L, N, retstep=True, endpoint=False)
    anal_field = np.zeros((N, 3))
    anal_field[:, 0] = -(2 * np.pi * np.cos(x * 2 * np.pi))

    charge_density_anal = ((2 * np.pi) ** 2 * np.sin(x * 2 * np.pi))

    g = Grid(L, NG, epsilon_0)
    indices_in_denser_grid = np.searchsorted(x, g.x)
    g.charge_density[1:-1] = charge_density_anal[indices_in_denser_grid]  # / resolution_increase

    energy_fourier = g.solve()
    energy_direct = g.direct_energy_calculation() * resolution_increase
    print("dx", dx, "fourier", energy_fourier, "direct", energy_direct, energy_fourier / energy_direct)

    def plots():
        fig, xspace = plt.subplots()
        xspace.set_title(
            r"Solving the Poisson equation $\Delta \psi = \rho / \epsilon_0$ via Fourier transform")
        xspace.plot(g.x, g.charge_density, "ro--", label=r"$\rho$")
        xspace.plot(x, charge_density_anal, "r-", lw=6, alpha=0.5, label=r"$\rho_a$")
        xspace.plot(g.x, g.electric_field, "bo--", alpha=0.5, label=r"$E$")
        xspace.plot(x, anal_field, "b-", lw=6, alpha=0.5, label=r"$E_a$")
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

    energy_correct = l2_test(energy_fourier, energy_direct)
    assert energy_correct, plots()
    field_correct = l2_test(g.electric_field[1:-1, 0], anal_field[indices_in_denser_grid][:, 0])
    assert field_correct, plots()


@pytest.mark.parametrize(["NG", "L"], [
    (128, 1),
    (128, 2 * np.pi)
    ])
def test_PoissonSolver_sheets(NG, L, debug=DEBUG, test_charge_density=1):
    epsilon_0 = 1

    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
    charge_density = np.zeros_like(x)
    region1 = (L * 1 / 8 < x) * (x < L * 2 / 8)
    region2 = (L * 5 / 8 < x) * (x < L * 6 / 8)
    charge_density[region1] = test_charge_density
    charge_density[region2] = -test_charge_density
    g = Grid(L, NG, epsilon_0)
    g.charge_density[1:-1] = charge_density
    g.solve()

    def plots():
        fig, axes = plt.subplots(3)
        ax0, ax1 = axes
        ax0.plot(x, charge_density)
        ax0.set_title("Charge density")
        ax1.set_title("Field")
        ax1.plot(x, g.electric_field, "r-")
        for ax in axes:
            ax.grid()
            ax.legend()
        plt.show()
        return "test_PoissonSolver_sheets failed!"

    if debug:
        plots()

    polynomial_coefficients = np.polyfit(x[region1], g.electric_field[1:-1, 0][region1], 1)
    first_bump_right = np.isclose(
        polynomial_coefficients[0], test_charge_density, rtol=1e-2)
    polynomial_coefficients = np.polyfit(x[region2], g.electric_field[1:-1, 0][region2], 1)
    second_bump_right = np.isclose(
        polynomial_coefficients[0], -test_charge_density, rtol=1e-2)
    assert first_bump_right and second_bump_right, plots()


@pytest.mark.parametrize(["NG", "L"], [
    (128, 1),
    (128, 2 * np.pi)
    ])
def test_PoissonSolver_ramp(NG, L, debug=DEBUG):
    """ For a charge density rho = Ax + B
    d2phi/dx2 = -rho/epsilon_0
    set epsilon_0 to 1
    d2phi/dx2 = Ax
    phi must be of form
    phi = -Ax^3/6 + Bx^2 + Cx + D"""

    a = 1

    # noinspection PyArgumentEqualDefault
    g = Grid(L, NG, epsilon_0=1)
    g.charge_density[1:-1] = a * g.x
    g.solve()
    field = a * (g.x - L / 2) ** 2 / 2

    def plots():
        fig, axes = plt.subplots(2)
        ax0, ax1 = axes
        ax0.plot(g.x, g.charge_density)
        ax0.set_title("Charge density")
        ax1.set_title("Field")
        ax1.plot(g.x, g.electric_field, "r-")
        ax1.plot(g.x, field, "g-")
        for ax in axes:
            ax.grid()
            ax.legend()
        plt.show()
        return "test_PoissonSolver_ramp failed!"

    if debug:
        plots()

    polynomial_coefficients = np.polyfit(g.x, g.electric_field[1:-1, 0], 2)
    assert np.isclose(polynomial_coefficients[0], a / 2, rtol=1e-2), (polynomial_coefficients[0], a / 2, plots())
