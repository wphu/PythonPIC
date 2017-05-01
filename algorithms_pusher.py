"""mathematical algorithms for the particle pusher, Leapfrog and Boris"""
# coding=utf-8
import numpy as np


def leapfrog_push(species, E, dt, *args):
    v_new = species.v[species.alive].copy()
    dv = E * species.q / species.m * dt
    v_new += dv
    energy = species.v[species.alive] * v_new * (0.5 * species.m)
    return species.x + v_new[:, 0] * dt, v_new, energy


def boris_push(species, E, dt, B):
    # add half electric impulse to v(t-dt/2)
    vminus = species.v + species.q * E / species.m * dt * 0.5

    # rotate to add magnetic field
    t = -B * species.q / species.m * dt * 0.5
    s = 2 * t / (1 + t * t)

    vprime = vminus + np.cross(vminus, t)  # TODO: axis?
    vplus = vminus + np.cross(vprime, s)
    v_new = vplus + species.q * E / species.m * dt * 0.5

    energy = species.v * v_new * (0.5 * species.m)
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


# @numba.njit()
def rela_boris_push(species, E: np.ndarray, dt: float, B: np.ndarray,
                    c: float = 1):
    """
    relativistic Boris pusher
    """
    vminus = species.v + species.q * E / species.m * dt * 0.5 # eq. 21 LPIC
    gamma_n = np.sqrt(1 + ((vminus / c) ** 2).sum(axis=1, keepdims=True))  # below eq 22 LPIC

    # rotate to add magnetic field
    t = B * species.q * dt / (2 * species.m * gamma_n)  # above eq 23 LPIC
    s = 2 * t / (1 + t * t)

    rot = rotation_matrix(t, s, species.N)

    vplus = np.einsum('ijk,ik->ij', rot, vminus)
    v_new = vplus + species.q * E / species.m * dt * 0.5

    # TODO: check correctness of relativistic kinetic energy calculation
    energy = 0.5 * species.m * (species.v * v_new).sum(axis=0)
    gamma_new = np.sqrt(1 + ((vminus / c) ** 2).sum(axis=1))
    x_new = species.x + v_new[:, 0] / gamma_new * dt
    return x_new, v_new, energy

if __name__ == "__main__":
    x = np.ones(10)
    v = np.ones((3, 10))
    E = np.zeros_like(v)
    B = np.zeros_like(E)
    q = 1
    m = 1
    dt = 1
    dx = 1
    c = 4

    xn, vn, e = rela_boris_push(x, v, E, B, q, m, dt, c)
