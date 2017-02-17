"""mathematical algorithms for the particle pusher, Leapfrog and Boris"""
# coding=utf-8
import numpy as np


# import numba

# @numba.njit() #TODO: add numba to this algorithm
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


# @numba.njit()
def rela_boris_push(x, v, E, B, q, m, dt, c=1):
    """
    relativistic Boris pusher
    """
    vminus = v + q * E / m * dt * 0.5
    n = x.size
    gamma_middle = np.sqrt(1 + ((vminus / c) ** 2).sum(axis=1, keepdims=True))

    # rotate to add magnetic field
    t = B * q * dt / (2 * m * gamma_middle)
    s = 2 * t / (1 + t * t)

    rot = rotation_matrix(t, s, n)

    vplus = np.einsum('ijk,ik->ij', rot, vminus)
    # import ipdb; ipdb.set_trace()

    v_new = vplus + q * E / m * dt * 0.5
    gamma_new = np.sqrt(1 + ((vminus / c) ** 2).sum(axis=1))
    # import ipdb; ipdb.set_trace()

    x_new = x + v_new[:, 0] / gamma_new * dt
    return x_new, v_new
