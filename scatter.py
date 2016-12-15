import numpy as np


def charge_density_deposition(x, dx, x_particles, particle_charge):
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
    charge_to_right = particle_charge * right_fractions
    charge_to_left = particle_charge * left_fractions
    charge_hist_to_right = np.roll(np.bincount(logical_coordinates, charge_to_right, minlength=x.size), +1)
    charge_hist_to_left = np.bincount(logical_coordinates, charge_to_left, minlength=x.size)
    return (charge_hist_to_right + charge_hist_to_left)

def current_density_deposition(x, dx, x_particles, particle_charge, velocity):
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
    current_hist = np.zeros((x.size, 3))
    logical_coordinates = (x_particles / dx).astype(int)
    right_fractions = (x_particles / dx - logical_coordinates).reshape(x_particles.size, 1)
    left_fractions = 1 - right_fractions
    current_to_right = particle_charge * velocity * right_fractions
    current_to_left = particle_charge * velocity * left_fractions
    # TODO: vectorise this instead of looping over dimensions
    for dim in range(3):
        current_hist[:,dim] += np.bincount(logical_coordinates, current_to_left[:,dim], minlength=x.size)
        current_hist[:,dim] += np.roll(np.bincount(logical_coordinates, current_to_right[:,dim], minlength=x.size), +1)
    return current_hist
