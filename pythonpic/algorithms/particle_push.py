# coding=utf-8
"""mathematical algorithms for the particle pusher, Leapfrog and Boris"""
import functools
import numpy as np
from numba import jit


def boris_push(species, E, dt, B):
    # add half electric impulse to v(t-dt/2)
    vminus = species.v + species.eff_q * E / species.eff_m * dt * 0.5

    # rotate to add magnetic field
    t = -B * species.eff_q / species.eff_m * dt * 0.5
    s = 2 * t / (1 + t * t)

    vprime = vminus + np.cross(vminus, t)
    vplus = vminus + np.cross(vprime, s)
    v_new = vplus + species.eff_q * E / species.eff_m * dt * 0.5

    energy = (species.v * v_new * (0.5 * species.eff_m)).sum()
    return species.x + v_new[:, 0] * dt, v_new, energy

@jit()
def rela_boris(x, v, c, eff_q, E, B, dt, eff_m):
    u = v / np.sqrt(1 - ((v ** 2).sum(axis=1, keepdims=True)) / c ** 2)  # below eq 22 LPIC
    half_force = eff_q * E / eff_m * dt * 0.5  # eq. 21 LPIC # array of shape (N_particles, 3)
    # add first half of electric force
    uminus = u + half_force

    # rotate to add magnetic field
    t = B * eff_q * dt / (2 * eff_m * np.sqrt(1 + ((uminus ** 2).sum(axis=1, keepdims=True) / c ** 2)))
    s = 2 * t / (1 + t * t)
    # u' = u- + u- x t
    uprime = uminus + np.cross(uminus, t)
    # v+ = u- + u' x s
    uplus = uminus + np.cross(uprime, s)

    # add second half of electric force
    u_new = uplus + half_force

    # CHECK: check correctness of relativistic kinetic energy calculation (needs to be at half timestep!)
    final_gamma = np.sqrt(1 + ((u_new ** 2).sum(axis=1, keepdims=True) / c ** 2))
    v_new = u_new / final_gamma
    # CHECK mean_gamma = (init_gamma + final_gamma)*0.5
    energy = ((final_gamma - 1) * eff_m * c ** 2).sum()
    x_new = x + v_new[:, 0] * dt
    return x_new, v_new, energy

def rela_boris_push(species, E: np.ndarray, dt: float, B: np.ndarray):
    """
    relativistic Boris pusher
    """
    return rela_boris(species.x, species.v, species.c, species.eff_q, E, B, dt, species.eff_m)

