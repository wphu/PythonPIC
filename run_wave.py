""" Run wave propagation"""
# coding=utf-8
import numpy as np

import plotting
from Constants import Constants
from Grid import Grid
from Simulation import Simulation
from Species import Species
from helper_functions import plotting_parser


def wave_propagation(filename,
                           bc,
                           bc_parameter_function,
                           bc_params,
                           save_data: bool = True,
                           ):
    """Implements wave propagation"""
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
    run.run(save_data)
    return run

if __name__ == '__main__':
    show, save, animate = plotting_parser("Wave propagation")
    s = wave_propagation("laser2", "laser", lambda t: t / 25, (1, 2))
    plotting.plotting(s, show=show, alpha=0.5, save=save, animate=animate)
    s = wave_propagation("laser6", "laser", lambda t: t / 25, (1, 6))
    plotting.plotting(s, show=show, alpha=0.5, save=save, animate=animate)
