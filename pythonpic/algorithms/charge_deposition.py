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
    logical_coordinates = (x_particles // dx).astype(int)
    left_fractions = x_particles / dx - logical_coordinates
    deposition_indices = np.copy(logical_coordinates)
    left_half = left_fractions < 0.5
    right_half = ~left_half
    deposition_indices[right_half] += 1
    deposition_indices[left_half] -= 1


    charge_to_current = 0.5 * np.ones_like(logical_coordinates)
    charge_to_other = 0.5 * np.ones_like(logical_coordinates)

    charge_to_current[left_half] += left_fractions[left_half]
    charge_to_other[left_half] -= left_fractions[left_half]

    charge_to_current[right_half] += 1- left_fractions[right_half]
    charge_to_other[right_half] -= 1+ left_fractions[right_half]

    charge_hist_to_right = np.bincount(logical_coordinates+1, charge_to_current, minlength=x.size+2)
    charge_hist_to_left = np.bincount(deposition_indices+1, charge_to_other, minlength=x.size+2)
    return charge_hist_to_right + charge_hist_to_left


def periodic_density_deposition(x, dx: float, x_particles):
    result = density_deposition(x, dx, x_particles)
    result[1] += result[-1]
    result[-2] += result[0]
    return result[1:]


def aperiodic_density_deposition(x, dx: float, x_particles):
    result = density_deposition(x, dx, x_particles)
    return result[1:]


