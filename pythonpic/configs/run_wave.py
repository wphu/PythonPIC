""" Run wave propagation"""
# coding=utf-8
import numpy as np

from pythonpic.algorithms import FieldSolver, BoundaryCondition
from pythonpic.algorithms.helper_functions import plotting_parser
from pythonpic.classes import Grid, Simulation
from pythonpic.visualization import plotting


def wave_propagation(filename,
                     bc = BoundaryCondition.non_periodic_bc(
                                               BoundaryCondition.Laser(1, 10, 3).laser_pulse),
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
    grid = Grid(T=T, L=L, NG=NG, epsilon_0=epsilon_0, bc=bc, solver=FieldSolver.BunemanSolver)
    alpha = c * grid.dt / grid.dx
    print(f"alpha is {alpha}")
    assert alpha <= 1
    description = \
        f"""Electrostatic wave driven by boundary condition
    """

    run = Simulation(grid, [], filename=filename, title=description)
    run.grid_species_initialization()
    run.run(save_data)
    return run


def main():
    args = plotting_parser("Weak beam instability")
    for filename, boundary_function in zip(["Wave", "Envelope", "Laser"],
                                           [BoundaryCondition.non_periodic_bc(
                                               BoundaryCondition.Laser(1, 10, 3).laser_wave),
                                            BoundaryCondition.non_periodic_bc(
                                                BoundaryCondition.Laser(1, 10, 3).laser_envelope),
                                            BoundaryCondition.non_periodic_bc(
                                                BoundaryCondition.Laser(1, 10, 3).laser_pulse),
                                            ]):
        s = wave_propagation(filename, boundary_function)
        plotting.plots(s, *args, alpha=0.5)


if __name__ == "__main__":
    main()
