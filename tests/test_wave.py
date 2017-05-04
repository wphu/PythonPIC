"""Tests the Leapfrog wave PDE solver"""
# coding=utf-8
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import pytest

from helper_functions import show_on_fail
# TODO: use of multigrid methods for wave equation
from plotting import plotting
from run_wave import wave_propagation


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


# @pytest.mark.parametrize(["NX", "NT", "c", "dx", "dt", "initial_field"],
#                          [(1000, 2000, 1 / 2, 0.01, 0.01, lambda x, NX, dx: np.sin(np.pi * x)),
#                           (1000, 200, 1 / 2, 0.01, 0.01,
#                            lambda x, NX, dx: np.exp(-(x - NX / 2 * dx) ** 2 / (0.1 * NX * dx))),
#                           ])
# def test_dAlambert(NX, NT, c, dx, dt, initial_field):
#     alpha = c * dt / dx
#     print(f"alpha is {alpha}")
#     assert alpha <= 1, f"alpha is {alpha}, may not be stable"
#
#     X = np.arange(NX) * dx
#     T = np.arange(NT) * dt
#     field = np.zeros(NX+2)
#     field[1:-1] = initial_field(X, NX, dx)
#     derivative = np.zeros_like(field)
#     field[0] = field[-1] = 0
#     field_first = LeapfrogWaveInitial(field, derivative, c, dx, dt)
#
#     field_history = np.zeros((NT, NX), dtype=float)
#     field_history[0] = field[1:-1]
#     field_history[1] = field_first
#     for n in range(2, NT):
#         field_history[n, 1:-1], _ = LeapfrogWaveSolver(field_history[n - 1], field_history[n - 2], c, dx, dt)
#     XGRID, TGRID = np.meshgrid(X, T)
#     analytical_solution = (initial_field(XGRID - c * TGRID, NX, dx) + initial_field(XGRID + c * TGRID, NX,
#                                                                                     dx)) / 2
#
#     assert l2_test(analytical_solution, field_history), plot_all(field_history, analytical_solution)
#     # assert False, plot_all(field_history, analytical_solution)
#
#
# def plots(T, boundary_value, measured_value_half, expected_value_half):
#     plt.plot(T, boundary_value, label="Boundary")
#     plt.plot(T, measured_value_half, label="Measured")
#     plt.plot(T, expected_value_half, label="Expected")
#     plt.legend()
#     plt.show()
#
#
# @pytest.mark.parametrize(["NX", "NT", "c", "dx", "dt", "boundary_condition"],
#                          [(100, 200000, 5, 0.01, 0.001, lambda n, dt, NT: sine_boundary_condition(n * dt, dt)),
#                           # (100, 200000, 5, 0.01, 0.001, lambda n, dt, NT: laser_boundary_condition(n * dt, NT*dt/2,
#                           #  NT*dt*1, 2)),
#                           # (100, 200000, 5, 0.01, 0.001, lambda n, dt, NT: laser_boundary_condition(n * dt, NT*dt/2,
#                           #  NT*dt*2, 2)),
#                           # (100, 200000, 5, 0.01, 0.001, lambda n, dt, NT: laser_boundary_condition(n * dt, NT*dt/2,
#                           #  NT*dt*3, 2)),
#                           # (100, 200000, 5, 0.01, 0.001, lambda n, dt, NT: laser_boundary_condition(n * dt, NT*dt/2,
#                           #  NT*dt*4, 2)),
#                           # (100, 200000, 5, 0.01, 0.001, lambda n, dt, NT: laser_boundary_condition(n * dt, NT*dt/2,
#                           #  NT*dt*10, 2)),
#                           ])
# def test_BC(NX, NT, c, dx, dt, boundary_condition):
#     alpha = c * dt / dx
#     print(f"alpha is {alpha}")
#     assert alpha <= 1, f"alpha is {alpha}, may not be stable"
#
#     X = np.arange(NX) * dx
#     T = np.arange(NT) * dt
#     field = np.zeros(NX+2)
#     derivative = np.zeros_like(field)
#     field[0] = boundary_condition(0, dt, NT)
#     field[-1] = 0
#     field_first = LeapfrogWaveInitial(field, derivative, c, dx, dt)
#     field_first[0] = boundary_condition(dt, dt, NT)
#
#     field_history = np.zeros((NT, NX), dtype=float)
#     field_history[0] = field[1:-1]
#     field_history[1] = field_first[1:-1]
#     for n in range(2, NT):
#         field_history[n, 1:-1], _ = LeapfrogWaveSolver(field_history[n - 1], field_history[n - 2], c, dx, dt)
#         field_history[n, 0] = boundary_condition(n * dt, dt, NT)
#
#     measured_value_half = field_history[:, int(NX / 2)]
#     expected_value_half = boundary_condition(T, dt, NT) / 2
#     assert l2_test(measured_value_half, expected_value_half), plots(T, field_history[:, 0], measured_value_half,
#                                                                     expected_value_half)


@pytest.mark.parametrize(["filename", "bc", "bc_parameter_function", "bc_params"],
                         [("sine", "sine", lambda T: 10, (0,)),
                          ("laser", "laser", lambda T: T / 25, (1, 2)),
                          ])
def test_wave_propagation(filename, bc, bc_parameter_function, bc_params):
    run = wave_propagation(filename, bc, bc_parameter_function, bc_params, save_data=False)
    assert run.grid.grid_energy_history.mean() > 0, plotting(run, show=show_on_fail, save=False, animate=True)

@pytest.mark.parametrize(["filename", "bc", "bc_parameter_function", "bc_params"],
                         [("sine_polarized", "sine", lambda T: 10, (0,)),
                          ("laser_polarized", "laser", lambda T: T / 25, (1, 2)),
                          ])
def test_polarization_orthogonality(filename, bc, bc_parameter_function, bc_params, plotting=False):
    # run = wave_propagation(filename, bc, bc_parameter_function, bc_params, polarization_angle, save_data=False)
    run = wave_propagation(filename, bc, bc_parameter_function, bc_params, save_data=False)
    angles = ((run.grid.electric_field_history[:,:,1:] * run.grid.magnetic_field_history).sum(axis=(1,2)))

    if plotting:
        i = int(run.NT-1)
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(131, projection='3d')
        # ax = fig.add_subplot(111, projection='3d')
        electric, = ax.plot(run.grid.x, run.grid.electric_field_history[i,1:-1,1], run.grid.electric_field_history[i,1:-1,2], "b-", label="E")
        magnetic, = ax.plot(run.grid.x, run.grid.magnetic_field_history[i,1:-1,0], run.grid.magnetic_field_history[i,1:-1,1], "g-", label="B")
        electric_bc, = ax.plot([0], run.grid.electric_field_history[i,0,1], run.grid.electric_field_history[i,0,2], "bo", label="E boundary")
        magnetic_bc, = ax.plot([0], run.grid.magnetic_field_history[i,0,0], run.grid.magnetic_field_history[i,0,1], "go", label="B boundary")
        ax.plot(run.grid.x, np.zeros_like(run.grid.x), np.zeros_like(run.grid.x), "k--")
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # ax.view_init(0,0)
        title = ax.set_title(f"{i}/{run.NT}")

        Fplus = 0.5 * (run.grid.electric_field_history[:, 1:-1, 1] + run.grid.c * run.grid.magnetic_field_history[:, 1:-1, 1]) # 0.5 * (E_y + cB_z)
        Fminus = 0.5 * (run.grid.electric_field_history[:, 1:-1, 1] - run.grid.c * run.grid.magnetic_field_history[:, 1:-1, 1]) # 0.5 * (E_y - cB_z)
        Gplus = 0.5 * (run.grid.electric_field_history[:, 1:-1, 2] + run.grid.c * run.grid.magnetic_field_history[:, 1:-1, 0]) # 0.5 * (E_z + cB_y)
        Gminus = 0.5 * (run.grid.electric_field_history[:, 1:-1, 2] - run.grid.c * run.grid.magnetic_field_history[:, 1:-1, 0]) # 0.5 * (E_z - cB_y)

        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        Fplus_plot, = ax2.plot(run.grid.x, Fplus[i], label="Fplus")
        Fminus_plot, = ax2.plot(run.grid.x, Fminus[i], label="Fminus")
        Gplus_plot, = ax2.plot(run.grid.x, Gplus[i], label="Gplus")
        Gminus_plot, = ax2.plot(run.grid.x, Gminus[i], label="Gminus")
        angles_plot, = ax3.plot(angles, label="angles")
        ax2.legend(loc='upper right')
        ax3.legend(loc='upper right')

        def animate(i):
            electric.set_data(run.grid.x, run.grid.electric_field_history[i,1:-1,1])
            electric.set_3d_properties(run.grid.electric_field_history[i,1:-1,2])
            magnetic.set_data(run.grid.x, run.grid.magnetic_field_history[i,1:-1,0])
            magnetic.set_3d_properties(run.grid.magnetic_field_history[i,1:-1,1])
            magnetic_bc.set_data(0, run.grid.magnetic_field_history[i,0,0])
            magnetic_bc.set_3d_properties(run.grid.magnetic_field_history[i,0,1])
            electric_bc.set_data(0, run.grid.electric_field_history[i,0,1])
            electric_bc.set_3d_properties(run.grid.electric_field_history[i,0,2])
            Fplus_plot.set_ydata(Fplus[i])
            Fminus_plot.set_ydata(Fminus[i])
            Gplus_plot.set_ydata(Gplus[i])
            Gminus_plot.set_ydata(Gminus[i])
            title.set_text(f"{i}/{run.NT}")
            return [electric, magnetic, electric_bc, magnetic_bc, Fplus_plot, Fminus_plot, Gplus_plot, Gminus_plot, title]
            # return [electric, magnetic, title]
        timestep = np.log(run.NT).astype(int)
        # anim = FuncAnimation(fig, animate, range(0, run.NT, timestep), interval=50)
        # anim.save("laser.mp4")
        plt.show()
    assert np.isclose(angles, 0).all(), "Polarization is not orthogonal!"

if __name__ == '__main__':
    test_polarization_orthogonality("sine_polarized", "sine", lambda T: 5, (0,), plotting=True)