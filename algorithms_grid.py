import numpy as np
from scipy import fftpack as fft


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


def interpolateField(x_particles, scalar_field, x, dx):
    """gathers field from grid to particles

    the reverse of the algorithm from charge_density_deposition

    there is no need to use numpy.bincount as the map is
    not N (number of particles) to M (grid), but M to N, N >> M
    """
    indices_on_grid = (x_particles / dx).astype(int)
    NG = scalar_field.size
    field = (x[indices_on_grid] + dx - x_particles) * scalar_field[indices_on_grid] +\
        (x_particles - x[indices_on_grid]) * scalar_field[(indices_on_grid + 1) % NG]
    return field / dx


def PoissonSolver(rho, k, NG, epsilon_0=1, neutralize=True):
    """solves the Poisson equation spectrally, via FFT

    the Poisson equation can be written either as
    (in position space)
    $$\nabla \cdot E = \rho/\epsilon_0$$
    $$\nabla^2 V = -\rho/\epsilon_0$$

    Assuming that all functions in fourier space can be represented as
    $$\exp{i(kx - \omega t)}$$
    It is easy to see that upon Fourier transformation $\nabla \to ik$, so

    (in fourier space)
    $$E = \rho /(ik \epsilon_0)$$
    $$V = \rho / (-k^2 \epsilon_0)$$

    Calculate that, fourier transform back to position space
    and both the field and potential pop out easily

    The conceptually problematic part is getting the $k$ wave vector right
    #TODO: finish this description
    """

    rho_F = fft.fft(rho)
    if neutralize:
        rho_F[0] = 0
    field_F = rho_F / (1j * k * epsilon_0)
    potential_F = field_F / (-1j * k * epsilon_0)
    field = fft.ifft(field_F).real
    # TODO: check for differences with finite difference field gotten from potential
    potential = fft.ifft(potential_F).real
    energy_presum = (rho_F * potential_F.conjugate()).real[:int(NG / 2)] / 2
    return field, potential, energy_presum