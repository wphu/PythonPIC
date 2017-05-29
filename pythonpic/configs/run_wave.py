""" Run wave propagation"""
# coding=utf-8
import numpy as np

from pythonpic.helper_functions.file_io import try_run
from pythonpic.algorithms import FieldSolver, BoundaryCondition
from pythonpic.algorithms.helper_functions import plotting_parser
from pythonpic.classes import Grid, Simulation
from pythonpic.visualization import plotting


def wave_propagation(filename,
                     bc = BoundaryCondition.Laser(1, 1, 10, 3).laser_pulse,
                     save_data: bool = True,
                     ):
    """Implements wave propagation"""
    T = 50
    NG = 60
    L = 2 * np.pi
    epsilon_0 = 1
    c = 1
    grid = Grid(T=T, L=L, NG=NG, epsilon_0=epsilon_0, bc=bc, periodic=False)
    description = \
        f"""Electrostatic wave driven by boundary condition
    """

    run = Simulation(grid, [], filename=filename, title=description)
    run.grid_species_initialization()
    run.run(save_data)
    return run


category_name = "wave"
def main():
    args = plotting_parser("Wave propagation")
    for filename, boundary_function in zip(["Wave", "Envelope", "Laser"],
                                           [BoundaryCondition.Laser(1, 1, 10, 3).laser_wave,
                                            BoundaryCondition.Laser(1, 1, 10, 3).laser_envelope,
                                            BoundaryCondition.Laser(1, 1, 10, 3).laser_pulse,
                                            ]):
        s = try_run(filename, category_name, wave_propagation, bc=boundary_function)
        plotting.plots(s, *args, alpha=0.5)


if __name__ == "__main__":
    main()
