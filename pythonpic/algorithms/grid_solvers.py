"""Numerical algorithms for the grid - interpolation to and from the grid, PDE solvers"""
# coding=utf-8
import numpy as np
from scipy import fftpack as fft


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
    # DOCUMENTATION: finish this description
    """

    rho_F = fft.fft(rho)
    if neutralize:
        rho_F[0] = 0
    field_F = rho_F / (1j * k * epsilon_0)
    potential_F = field_F / (-1j * k * epsilon_0)
    field = fft.ifft(field_F).real
    # TODO: check for differences with finite difference field gotten from potential
    energy_presum = (rho_F * potential_F.conjugate()).real[:int(NG / 2)] / 2
    return field, energy_presum


def BunemanWaveSolver(electric_field, magnetic_field, current_x, current_yz, dt, dx, c, epsilon_0):
    # dt = dx/c
    Fplus = 0.5 * (electric_field[:, 1] + c * magnetic_field[:, 2])
    Fminus = 0.5 * (electric_field[:, 1] - c * magnetic_field[:, 2])
    Gplus = 0.5 * (electric_field[:, 2] + c * magnetic_field[:, 1])
    Gminus = 0.5 * (electric_field[:, 2] - c * magnetic_field[:, 1])

    Fplus[1:] = Fplus[:-1] - 0.5 * dt * (current_yz[2:-1, 0]) / epsilon_0
    Fminus[:-1] = Fminus[1:] - 0.5 * dt * (current_yz[1:-2, 0]) / epsilon_0  # TODO: verify the index on current here
    Gplus[1:] = Gplus[:-1] - 0.5 * dt * (current_yz[2:-1, 1]) / epsilon_0
    Gminus[:-1] = Gminus[1:] - 0.5 * dt * (current_yz[1:-2, 1]) / epsilon_0  # TODO: verify the index on current here

    new_electric_field = np.zeros_like(electric_field)
    new_magnetic_field = np.zeros_like(magnetic_field)

    new_electric_field[:, 1] = Fplus + Fminus
    new_electric_field[:, 2] = Gplus + Gminus
    new_magnetic_field[:, 1] = (Gplus - Gminus) / c
    new_magnetic_field[:, 2] = (Fplus - Fminus) / c

    new_electric_field[:, 0] = electric_field[:, 0] - dt / epsilon_0 * current_x[:-1]  # TODO: verify indices here
    electric_energy = 0.5 * epsilon_0 * dx * (new_electric_field ** 2).sum()
    magnetic_energy = 0.5 * dx * (new_magnetic_field ** 2).sum()
    return new_electric_field, new_magnetic_field, electric_energy + magnetic_energy
