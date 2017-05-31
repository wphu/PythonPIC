""" Run wave propagation"""
# coding=utf-8
import numpy as np

from ..algorithms import BoundaryCondition
from ..classes import Grid, Simulation


def wave_propagation(filename,
                     bc = BoundaryCondition.Laser(1, 1, 10, 3).laser_pulse,
                     ):
    """Implements wave propagation"""
    T = 50
    NG = 60
    L = 2 * np.pi
    epsilon_0 = 1
    c = 1
    grid = Grid(T=T, L=L, NG=NG, epsilon_0=epsilon_0, c=c, bc=bc, periodic=False)
    description = "Electrostatic wave driven by boundary condition"

    run = Simulation(grid, [], filename=filename, category_type="wave", title=description)
    return run


