# coding=utf-8
"""mathematical algorithms for the particle pusher, Leapfrog and Boris"""
import functools

# coding=utf-8
import numpy as np

from pythonpic.algorithms.helper_functions import gamma_from_v, gamma_from_u


def boris_push(species, E, dt, B):
    # add half electric impulse to v(t-dt/2)
    vminus = species.v + species.q * E / species.m * dt * 0.5

    # rotate to add magnetic field
    t = -B * species.q / species.m * dt * 0.5
    s = 2 * t / (1 + t * t)

    vprime = vminus + np.cross(vminus, t)  # TODO: axis?
    vplus = vminus + np.cross(vprime, s)
    v_new = vplus + species.q * E / species.m * dt * 0.5

    energy = (species.v * v_new * (0.5 * species.m)).sum()
    return species.x + v_new[:, 0] * dt, v_new, energy


# @numba.njit() # OPTIMIZE: add numba to this algorithm
def rotation_matrix(t: np.ndarray, s: np.ndarray, n: int) -> np.ndarray:
    """
    Implements the heavy lifting rotation matrix calculation of the Boris pusher
    Parameters
    ----------
    t (np.ndarray): vector
    s (np.ndarray): vector
    n (int): number of particles

    Returns
    -------
    result (np.ndarray): rotation matrix
    """
    result = np.zeros((n, 3, 3))
    result[:] = np.eye(3)

    sz = s[:, 2]
    sy = s[:, 1]
    tz = t[:, 2]
    ty = t[:, 1]
    sztz = sz * tz
    syty = sy * ty
    result[:, 0, 0] -= sztz
    result[:, 0, 0] -= syty
    result[:, 0, 1] = sz
    result[:, 1, 0] = -sz
    result[:, 0, 2] = -sy
    result[:, 2, 0] = sy
    result[:, 1, 1] -= sztz
    result[:, 2, 2] -= syty
    result[:, 2, 1] = sy * tz
    result[:, 1, 2] = sz * ty
    return result


def lpic_solve(t, s, N, uminus):
    rot = rotation_matrix(t, s, N)  # array of shape (3,3) for each of N_particles, so (N_particles, 3, 3)
    uplus = np.einsum('ijk,ik->ij', rot, uminus)
    return uplus


def bl_solve(t, s, N, uminus):
    # u' = u- + u- x t 
    uprime = uminus + np.cross(uminus, t)
    # v+ = u- + u' x s
    uplus = uminus + np.cross(uprime, s)
    return uplus


# @numba.njit()
def rela_boris_push(species, E: np.ndarray, dt: float, B: np.ndarray,
                    solve=lpic_solve):
    """
    relativistic Boris pusher
    """
    u = species.v * gamma_from_v(species.v, species.c)
    half_force = species.q * E / species.m * dt * 0.5  # eq. 21 LPIC # array of shape (N_particles, 3)
    # add first half of electric force
    uminus = u + half_force

    # rotate to add magnetic field
    t = B * species.q * dt / (2 * species.m * gamma_from_u(uminus, species.c))  # above eq 23 LPIC
    s = 2 * t / (1 + t * t)
    uplus = solve(t, s, species.N, uminus)

    # add second half of electric force
    u_new = uplus + half_force

    # TODO: check correctness of relativistic kinetic energy calculation
    #   import ipdb; ipdb.set_trace()
    final_gamma = gamma_from_u(u_new, species.c)
    v_new = u_new / final_gamma
    energy = ((final_gamma - 1) * species.m * species.c ** 2).sum()
    x_new = species.x + v_new[:, 0] * dt
    return x_new, v_new, energy


rela_boris_push_bl = functools.partial(rela_boris_push, solve=bl_solve)
rela_boris_push_lpic = functools.partial(rela_boris_push, solve=lpic_solve)
