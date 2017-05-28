# coding=utf-8
import functools

from pythonpic.algorithms import grid_solvers



class Solver:
    def __init__(self, solve_algorithm, initialiation_algorithm):
        self.solve = solve_algorithm
        self.init_solver = initialiation_algorithm


def solve_fourier(grid, neutralize = False):
    grid.electric_field[1:-1, 0], grid.energy_per_mode = grid_solvers.PoissonSolver(
        grid.charge_density[:-1], grid.k, grid.NG, epsilon_0=grid.epsilon_0, neutralize=neutralize
        )
    return grid.energy_per_mode.sum() / (grid.NG / 2)  # * 8 * np.pi * grid.k[1]**2


solve_fourier_neutral = functools.partial(solve_fourier, neutralize=True)


def solve_buneman(grid):
    grid.electric_field, grid.magnetic_field, grid.energy_per_mode = grid_solvers.BunemanWaveSolver(
        grid.electric_field, grid.magnetic_field, grid.current_density_x, grid.current_density_yz, grid.dt, grid.dx, grid.c, grid.epsilon_0)
    return grid.energy_per_mode


FourierSolver = Solver(solve_fourier_neutral, solve_fourier_neutral)
BunemanSolver = Solver(solve_buneman, solve_fourier)
