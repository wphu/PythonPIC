# Algorithms for deposition of charge and currents on grid
# coding=utf-8
import numpy as np


def density_deposition(x, dx: float, x_particles):
    """scatters charge from particles to grid
    uses linear interpolation
    x_i | __________p___| x_i+1
    for a particle $p$ in cell $i$ of width $dx$ the location in cell is defined as
    $$X_p = x_p - x_i$$
    then, $F_r = X_p/dx$ is the fraction of charge going to the right side of the cell
    (as $X_p \to dx$, the particle is closer to the right cell)
    while $F_l = 1-F_r$ is the fraction of charge going to the left

    numpy.bincount is used to count particles in cells
    the weights are the fractions for each cell

    to change the index for the right going  (keeping periodic boundary conditions)
    numpy.roll is used
    """
    logical_coordinates = (x_particles / dx).astype(int)
    right_fractions = x_particles / dx - logical_coordinates
    left_fractions = 1 - right_fractions
    charge_to_right = right_fractions
    charge_to_left = left_fractions
    charge_hist_to_right = np.bincount(logical_coordinates+1, charge_to_right, minlength=x.size+1)
    charge_hist_to_left = np.bincount(logical_coordinates, charge_to_left, minlength=x.size+1)
    return charge_hist_to_right + charge_hist_to_left


def periodic_density_deposition(x, dx: float, x_particles):
    result = density_deposition(x, dx, x_particles)
    result[0] += result[-1]
    return result


def aperiodic_density_deposition(x, dx: float, x_particles, particle_charge: float):
    result = density_deposition(x, dx, x_particles)
    return result


def interpolateField(x_particles, scalar_field, x, dx: float):
    """gathers field from grid to particles

    the reverse of the algorithm from charge_density_deposition

    there is no need to use numpy.bincount as the map is
    not N (number of particles) to M (grid), but M to N, N >> M
    """
    logical_coordinates = (x_particles / dx).astype(int)
    NG = scalar_field.size
    right_fractions = x_particles / dx - logical_coordinates
    field = (1 - right_fractions) * scalar_field[logical_coordinates] + \
            right_fractions * scalar_field[(logical_coordinates + 1) % NG]
    return field
