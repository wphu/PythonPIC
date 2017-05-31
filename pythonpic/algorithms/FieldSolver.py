# coding=utf-8
import functools

from ..algorithms.grid_solvers import PoissonLongitudinalSolver, BunemanTransversalSolver, BunemanLongitudinalSolver



class Solver:
    def __init__(self, solve_algorithm, initialiation_algorithm):
        self.solve = solve_algorithm
        self.init_solver = initialiation_algorithm


def solve_fourier(grid, neutralize = False):
    grid.electric_field[1:-1, 0] = PoissonLongitudinalSolver(
        grid.charge_density[:-1], grid.k, grid.NG, epsilon_0=grid.epsilon_0, neutralize=neutralize
        )

    grid.electric_field[:, 1:], grid.magnetic_field[:, 1:] = BunemanTransversalSolver(grid.electric_field[:, 1:],
                                                                                      grid.magnetic_field[:, 1:],
                                                                                      grid.current_density_yz, grid.dt,
                                                                                      grid.c, grid.epsilon_0)
    return None


solve_fourier_neutral = functools.partial(solve_fourier, neutralize=True)


def solve_buneman(grid):
    grid.electric_field[:, 0] = BunemanLongitudinalSolver(grid.electric_field[:, 0],
                                                          grid.current_density_x,
                                                          grid.dt,
                                                          grid.epsilon_0,
                                                          )
    grid.electric_field[:, 1:], grid.magnetic_field[:, 1:] = BunemanTransversalSolver(grid.electric_field[:, 1:],
                                                                                      grid.magnetic_field[:, 1:],
                                                                                      grid.current_density_yz, grid.dt,
                                                                                      grid.c, grid.epsilon_0)
    return None


FourierSolver = Solver(solve_fourier_neutral, solve_fourier_neutral)
BunemanSolver = Solver(solve_buneman, solve_fourier)
