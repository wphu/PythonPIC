""" Run wave propagation"""
# coding=utf-8
import numpy as np

import plotting
from Constants import Constants
from Grid import Grid
from Simulation import Simulation
from helper_functions import plotting_parser


def wave_propagation(filename,
                           bc,
                           bc_parameter_function,
                           bc_params,
                           polarization_angle: float = 0,
                           save_data: bool = True,
                           ):
    """Implements wave propagation"""
    filename = f"data_analysis/EMWAVE/{filename}/{filename}.hdf5"
    T = 20
    print(f"T is {T}")
    NG = 360
    L = 2 * np.pi
    dx = L / (NG)
    epsilon_0 = 1
    c = 1
    dt = dx / c
    NT = np.ceil(T / dt).astype(int)
    grid = Grid(L, NG, epsilon_0, NT, dt=dt, n_species=0, solver="direct", bc=bc,
                bc_params=(bc_parameter_function(T), *bc_params), polarization_angle=polarization_angle)
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
    s = wave_propagation("laser2", "laser", lambda t: t / 3, (1, 2), 0)
    plotting.plotting(s, show=show, alpha=0.5, save=save, animate=animate)
    s = wave_propagation("laser6", "laser", lambda t: t / 3, (1, 6), 2*np.pi/3)
    plotting.plotting(s, show=show, alpha=0.5, save=save, animate=animate)
    s = wave_propagation("sine1", "sine", lambda t: 1, (1,), np.pi/4)
    plotting.plotting(s, show=show, alpha=0.5, save=save, animate=animate)
