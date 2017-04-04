"""Tests the Leapfrog wave PDE solver"""
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pytest

from Constants import Constants
from Grid import Grid
from Simulation import Simulation
from algorithms_grid import LeapfrogWaveSolver, LeapfrogWaveInitial, sine_boundary_condition
# TODO: use of multigrid methods for wave equation
from helper_functions import l2_test
from plotting import plotting


def plot_all(field_history, analytical_solution):
    T, X = field_history.shape
    XGRID, TGRID = np.meshgrid(np.arange(X), np.arange(T))
    for n in range(field_history.shape[0]):
        if field_history.shape[0] < 200 or n % 100:
            plt.plot(field_history[n])
    fig = plt.figure()
    ax = fig.add_subplot(211)
    CF1 = ax.contourf(TGRID, XGRID, field_history, alpha=1)
    ax.set_xlabel("time")
    ax.set_ylabel("space")
    plt.colorbar(CF1)
    ax2 = fig.add_subplot(212)
    CF2 = ax2.contourf(TGRID, XGRID, analytical_solution, alpha=1)
    ax2.set_xlabel("time")
    ax2.set_ylabel("space")
    plt.colorbar(CF2)
    plt.show()


@pytest.mark.parametrize(["NX", "NT", "c", "dx", "dt", "initial_field"],
                         [(1000, 2000, 1 / 2, 0.01, 0.01, lambda x, NX, dx: np.sin(np.pi * x)),
                          (1000, 200, 1 / 2, 0.01, 0.01,
                           lambda x, NX, dx: np.exp(-(x - NX / 2 * dx) ** 2 / (0.1 * NX * dx))),
                          ])
def test_dAlambert(NX, NT, c, dx, dt, initial_field):
    alpha = c * dt / dx
    print(f"alpha is {alpha}")
    assert alpha <= 1, f"alpha is {alpha}, may not be stable"

    X = np.arange(NX) * dx
    T = np.arange(NT) * dt
    field = initial_field(X, NX, dx)
    derivative = np.zeros_like(X)
    field[0] = field[-1] = 0
    field_first = LeapfrogWaveInitial(field, derivative, c, dx, dt)

    field_history = np.zeros((NT, NX), dtype=float)
    field_history[0] = field
    field_history[1] = field_first
    for n in range(2, NT):
        field_history[n], _ = LeapfrogWaveSolver(field_history[n - 1], field_history[n - 2], c, dx, dt)
    XGRID, TGRID = np.meshgrid(X, T)
    analytical_solution = (initial_field(XGRID - c * TGRID, NX, dx) + initial_field(XGRID + c * TGRID, NX,
                                                                                    dx)) / 2

    assert l2_test(analytical_solution, field_history), plot_all(field_history, analytical_solution)
    # assert False, plot_all(field_history, analytical_solution)


def plots(T, boundary_value, measured_value_half, expected_value_half):
    plt.plot(T, boundary_value, label="Boundary")
    plt.plot(T, measured_value_half, label="Measured")
    plt.plot(T, expected_value_half, label="Expected")
    plt.legend()
    plt.show()


@pytest.mark.parametrize(["NX", "NT", "c", "dx", "dt", "boundary_condition"],
                         [(100, 200000, 5, 0.01, 0.001, lambda n, dt, NT: sine_boundary_condition(n * dt, dt)),
                          # (100, 200000, 5, 0.01, 0.001, lambda n, dt, NT: laser_boundary_condition(n * dt, NT*dt/2,
                          #  NT*dt*1, 2)),
                          # (100, 200000, 5, 0.01, 0.001, lambda n, dt, NT: laser_boundary_condition(n * dt, NT*dt/2,
                          #  NT*dt*2, 2)),
                          # (100, 200000, 5, 0.01, 0.001, lambda n, dt, NT: laser_boundary_condition(n * dt, NT*dt/2,
                          #  NT*dt*3, 2)),
                          # (100, 200000, 5, 0.01, 0.001, lambda n, dt, NT: laser_boundary_condition(n * dt, NT*dt/2,
                          #  NT*dt*4, 2)),
                          # (100, 200000, 5, 0.01, 0.001, lambda n, dt, NT: laser_boundary_condition(n * dt, NT*dt/2,
                          #  NT*dt*10, 2)),
                          ])
def test_BC(NX, NT, c, dx, dt, boundary_condition):
    alpha = c * dt / dx
    print(f"alpha is {alpha}")
    assert alpha <= 1, f"alpha is {alpha}, may not be stable"

    X = np.arange(NX) * dx
    T = np.arange(NT) * dt
    field = np.zeros_like(X)
    derivative = np.zeros_like(X)
    field[0] = boundary_condition(0, dt, NT)
    field[-1] = 0
    field_first = LeapfrogWaveInitial(field, derivative, c, dx, dt)
    field_first[0] = boundary_condition(dt, dt, NT)

    field_history = np.zeros((NT, NX), dtype=float)
    field_history[0] = field
    field_history[1] = field_first
    for n in range(2, NT):
        field_history[n], _ = LeapfrogWaveSolver(field_history[n - 1], field_history[n - 2], c, dx, dt)
        field_history[n, 0] = boundary_condition(n * dt, dt, NT)

    measured_value_half = field_history[:, int(NX / 2)]
    expected_value_half = boundary_condition(T, dt, NT) / 2
    assert l2_test(measured_value_half, expected_value_half), plots(T, field_history[:, 0], measured_value_half,
                                                                    expected_value_half)


@pytest.mark.parametrize(["filename", "bc", "bc_parameter_function", "bc_params"],
                         [("sine", "sine", lambda T: 10, (0,)),
                          ("laser", "laser", lambda T: T / 25, (1, 2)),
                          ])
def test_Simulation(filename, bc, bc_parameter_function, bc_params):
    filename = f"data_analysis/EMWAVE/{filename}/{filename}.hdf5"
    NT = 2000
    dt = 0.01
    T = NT * dt
    print(f"T is {T}")
    NG = 100
    L = 2 * np.pi
    epsilon_0 = 1
    c = 1
    grid = Grid(L, NG, epsilon_0, NT, dt=dt, solver="direct", bc=bc, bc_params=(bc_parameter_function(T), *bc_params))
    alpha = c * dt / grid.dx
    print(f"alpha is {alpha}")
    assert alpha <= 1
    description = "Electrostatic wave driven by boundary condition\n"

    run = Simulation(NT, dt, Constants(c, epsilon_0), grid, [], filename=filename, title=description)
    run.grid_species_initialization()
    run.run()
    # plotting(run, show=True, save=True, animate=True)
    assert run.grid.grid_energy_history.mean() > 0, plotting(run, show=True, save=False, animate=True)


if __name__ == '__main__':
    test_Simulation()
