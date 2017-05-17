"""Tests the Leapfrog wave PDE solver"""
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pythonpic.algorithms import BoundaryCondition
from pythonpic.algorithms.helper_functions import show_on_fail
from pythonpic.configs.run_wave import wave_propagation
from pythonpic.visualization.plotting import plots


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


@pytest.mark.parametrize(["filename", "bc"],
                         [("sine", BoundaryCondition.non_periodic_bc(BoundaryCondition.Laser(1, 10, 3).laser_wave)),
                          ("envelope",
                           BoundaryCondition.non_periodic_bc(BoundaryCondition.Laser(1, 10, 3).laser_envelope)),
                          ("laser", BoundaryCondition.non_periodic_bc(BoundaryCondition.Laser(1, 10, 3).laser_pulse)),
                          ])
def test_wave_propagation(filename, bc):
    run = wave_propagation(filename, bc)
    assert run.grid.grid_energy_history.mean() > 0, plots(run, show=show_on_fail, save=False, animate=True)


@pytest.mark.parametrize(["filename", "bc"],
                         [("sine", BoundaryCondition.non_periodic_bc(BoundaryCondition.Laser(1, 10, 3).laser_wave)),
                          ("envelope",
                           BoundaryCondition.non_periodic_bc(BoundaryCondition.Laser(1, 10, 3).laser_envelope)),
                          ("laser", BoundaryCondition.non_periodic_bc(BoundaryCondition.Laser(1, 10, 3).laser_pulse)),
                          ])
def test_polarization_orthogonality(filename, bc):
    run = wave_propagation(filename, bc)
    angles = ((run.grid.electric_field_history[:, :, 1:] * run.grid.magnetic_field_history).sum(axis=(1, 2)))
    assert np.isclose(angles, 0).all(), "Polarization is not orthogonal!"
