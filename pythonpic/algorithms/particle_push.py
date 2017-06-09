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
def rela_boris_velocity_kick(v, c, eff_q, E, B, dt, eff_m):
    # calculate u
    v /= np.sqrt(1 - ((v ** 2).sum(axis=1, keepdims=True)) / c ** 2)  # below eq 22 LPIC
    half_force = (eff_q * 0.5 / eff_m * dt) * E  # eq. 21 LPIC # array of shape (N_particles, 3)
    # add first half of electric force

    #calculate uminus
    v += half_force

    # rotate to add magnetic field
    t = B * eff_q * dt / (2 * eff_m * np.sqrt(1 + ((v ** 2).sum(axis=1, keepdims=True) / c ** 2)))
    # u' = u- + u- x t
    uprime = v + np.cross(v, t)
    # v+ = u- + u' x s
    t *= 2
    t /= (1 + t * t)
    v += np.cross(uprime, t)

    # add second half of electric force
    v += half_force

    final_gamma = np.sqrt(1 + ((v ** 2).sum(axis=1, keepdims=True) / c ** 2))
    v /= final_gamma
    total_velocity = np.sqrt((v**2).sum(axis=1, keepdims=True))
    total_velocity *= final_gamma - 1
    return total_velocity.sum() * dt * eff_m * c ** 2


def rela_boris_push(species, E: np.ndarray, dt: float, B: np.ndarray):
    """
    relativistic Boris pusher
    """
    energy = rela_boris_velocity_kick(species.v, species.c, species.eff_q,
            E, B,
            dt, species.eff_m)
    new_x = species.x + species.v[:,0] * dt
    return new_x, species.v, energy

