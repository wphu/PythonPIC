import matplotlib.pyplot as plt
import numpy as np
import pytest

from algorithms_grid import DirectWaveSolver
# TODO: use of multigrid methods for wave equation
from helper_functions import l2_test


def sine_boundary_condition(t, dt, NT):
    return np.sin(t / 10 / NT / dt * 2 * np.pi)


@pytest.mark.parametrize(["NX", "NT", "c", "dx", "dt"], [(100, 100, 1, 0.2, 0.2)])
def test_convergence(NX, NT, c, dx, dt):
    alpha = c * dt / dx
    print(f"alpha is {alpha}")
    X = np.arange(NX) * dx
    T = np.arange(NT) * dt

    potential = np.exp(-(X - NX / 2 * dx) ** 2 / (1 / 10 * NX * dx) ** 2)
    potential[0] = potential[-1] = 0
    potential_first = np.zeros_like(potential)
    derivative = np.zeros_like(potential)
   
    potential_first[1:-1] = dt * derivative[1:-1] + \
                            potential[1:-1] * (1 - alpha ** 2) + \
                            0.5 * alpha ** 2 * (potential[:-2] + potential[2:])
    potential_history = np.zeros((NT, NX), dtype=float)
    potential_history[0] = potential
    potential_history[1] = potential_first
    for n in range(2, NT):
        potential_history[n] = DirectWaveSolver(potential_history[n - 1], potential_history[n - 2], c, dx, dt)
        plt.plot(X, potential_history[n])
    XGRID, TGRID = np.meshgrid(X, T)
    # analytical_solution = np.cos(XGRID*np.pi) * np.cos(alpha**2*np.pi * TGRID)
    fig = plt.figure()
    # ax = fig.add_subplot(211, projection='3d')
    # ax.plot_surface(TGRID, XGRID, potential_history, alpha=1)
    ax = fig.add_subplot(211)
    CF1 = ax.contourf(TGRID, XGRID, potential_history, alpha=1)
    ax.set_xlabel("time")
    ax.set_ylabel("space")
    # ax2 = fig.add_subplot(212, projection='3d')
    # ax2.plot_surface(TGRID, XGRID, analytical_solution, alpha=1)
    plt.colorbar(CF1)
    # ax2 = fig.add_subplot(212)
    # CF2 = ax2.contourf(TGRID, XGRID, analytical_solution, alpha=1)
    # ax2.set_xlabel("time")
    # ax2.set_ylabel("space")
    # plt.colorbar(CF2)
    plt.show()
    assert l2_test(analytical_solution, potential_history)

if __name__ == "__main__":
    L = 10
    dx = 0.1
    NX = np.ceil(L / dx).astype(int)
    dt = 0.05

    c = 1 / 4
    NT = 20
    test_convergence(NX, NT, c, dt, dx)
