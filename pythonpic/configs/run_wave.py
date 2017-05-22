""" Run wave propagation"""
# coding=utf-8
import numpy as np

from ..algorithms import FieldSolver, BoundaryCondition
from ..algorithms.helper_functions import plotting_parser, Constants
from ..classes import Grid
from ..classes import Simulation
from ..visualization import plotting


def wave_propagation(filename,
                     bc,
                     save_data: bool = True,
                     ):
    """Implements wave propagation"""
    filename = f"data_analysis/EMWAVE/{filename}/{filename}.hdf5"
    T = 50
    print(f"T is {T}")
    NG = 60
    L = 2 * np.pi
    epsilon_0 = 1
    c = 1
    grid = Grid(L, NG, epsilon_0, T=T, solver=FieldSolver.BunemanSolver, bc=bc)
    alpha = c * grid.dt / grid.dx
    print(f"alpha is {alpha}")
    assert alpha <= 1
    description = \
        f"""Electrostatic wave driven by boundary condition
    """

    run = Simulation(grid.NT, grid.dt, [], grid, Constants(c, epsilon_0), boundary_condition=bc, filename=filename,
                     title=description)
    run.grid_species_initialization()
    run.run(save_data)
    return run


def main():
    show, save, animate = plotting_parser("Wave propagation")
    for filename, boundary_function in zip(["Wave", "Envelope", "Laser"],
                                           [BoundaryCondition.non_periodic_bc(
                                               BoundaryCondition.Laser(1, 10, 3).laser_wave),
                                            BoundaryCondition.non_periodic_bc(
                                                BoundaryCondition.Laser(1, 10, 3).laser_envelope),
                                            BoundaryCondition.non_periodic_bc(
                                                BoundaryCondition.Laser(1, 10, 3).laser_pulse),
                                            ]):
        s = wave_propagation(filename, boundary_function)
        plotting.plots(s, show=show, alpha=0.5, save=save, animate=animate)


if __name__ == "__main__":
    main()
