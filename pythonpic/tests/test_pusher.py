# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ..algorithms import particle_push, helper_functions
from ..classes import Species, TimelessGrid, Frame

atol = 1e-1
rtol = 1e-4


@pytest.fixture(params=[1, 2, 3, 10, 100])
def _N_particles(request):
    return request.param


@pytest.fixture(params=[particle_push.boris_push])
def _pusher(request):
    return request.param


@pytest.fixture(params=[particle_push.rela_boris_push_lpic, particle_push.rela_boris_push_bl])
def _rela_pusher(request):
    return request.param


@pytest.fixture(params=np.linspace(0.1, 0.9, 10))
def _v0(request):
    return request.param


def plot(pusher, t, analytical_result, simulation_result,
         message="Difference between analytical and simulated results!"):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle(pusher)
    ax1.plot(t, analytical_result, "b-", label="analytical result")
    ax1.plot(t, simulation_result, "ro--", label="simulation result")

    ax2.plot(t, np.abs(simulation_result - analytical_result), label="difference")
    ax2.plot(t, np.ones_like(t) * (atol + rtol * np.abs(analytical_result)))

    ax2.set_xlabel("t")
    ax1.set_ylabel("result")
    ax2.set_ylabel("difference")
    for ax in [ax1, ax2]:
        ax.grid()
        ax.legend()
        ax.set_xticks(t)
    plt.show()
    return message


def test_constant_field(_pusher, _N_particles):
    T = 10
    dt = 10 / 200
    NT = helper_functions.calculate_number_timesteps(T, dt)
    g = Frame(dt, 1, 1, NT)
    s = Species(1, 1, _N_particles, g, pusher=_pusher)
    t = np.arange(0, T, dt * s.save_every_n_iterations) - dt / 2

    def uniform_field(x):
        return np.array([[1, 0, 0]], dtype=float)

    x_analytical = 0.5 * (t + dt / 2) ** 2 + 0
    s.init_push(uniform_field)
    for i in range(NT):
        s.save_particle_values(i)
        s.push(uniform_field)

    assert np.allclose(s.position_history[:, 0], x_analytical, atol=atol, rtol=rtol), \
        plot(_rela_pusher, t, x_analytical, s.position_history[:, 0])


# noinspection PyUnresolvedReferences
def test_relativistic_constant_field(_rela_pusher, _N_particles):
    T = 10
    dt = 10 / 200
    NT = helper_functions.calculate_number_timesteps(T, dt)
    g = Frame(dt, 1, 1, NT)
    s = Species(1, 1, _N_particles, g, pusher=_rela_pusher)
    t = np.arange(0, T, dt * s.save_every_n_iterations) - dt / 2

    def uniform_field(x):
        return np.array([[1, 0, 0]], dtype=float)

    v_analytical = (t - dt / 2) / np.sqrt((t - dt / 2) ** 2 + 1)
    s.init_push(uniform_field)
    for i in range(NT):
        s.save_particle_values(i)
        s.push(uniform_field)

    assert (s.velocity_history < 1).all(), plot(_rela_pusher, t, v_analytical, s.velocity_history[:, 0, 0],
                                                f"Velocity went over c! Max velocity: {s.velocity_history.max()}")
    assert np.allclose(s.velocity_history[:, 0, 0], v_analytical, atol=atol, rtol=rtol), \
        plot(_rela_pusher, t, v_analytical, s.velocity_history[:, 0, 0], )


# noinspection PyUnresolvedReferences
def test_relativistic_magnetic_field(_rela_pusher, _N_particles, _v0):
    B0 = 1
    T = 10
    dt = T / 200
    NT = helper_functions.calculate_number_timesteps(T, dt)
    g = Frame(dt, 1, 1, NT)
    s = Species(1, 1, _N_particles, g, pusher=_rela_pusher)
    t = np.arange(0, T, dt * s.save_every_n_iterations) - dt / 2
    s.v[:, 1] = _v0

    def uniform_magnetic_field(x):
        return np.array([[0, 0, B0]], dtype=float)

    def uniform_electric_field(x):
        return np.zeros(3, dtype=float)

    gamma = helper_functions.gamma_from_v(s.v, s.c)[0]
    vy_analytical = _v0 * np.cos(s.q * B0 * (t - dt / 2) / (s.m * gamma))

    s.init_push(uniform_electric_field, uniform_magnetic_field)
    for i in range(NT):
        s.save_particle_values(i)
        s.push(uniform_electric_field, uniform_magnetic_field)
    assert (s.velocity_history < 1).all(), plot(_rela_pusher, t, vy_analytical, s.velocity_history[:, 0, 1],
                                                f"Velocity went over c! Max velocity: {s.velocity_history.max()}")
    assert np.allclose(s.velocity_history[:, 0, 1], vy_analytical, atol=atol, rtol=rtol), \
        plot(_rela_pusher, t, vy_analytical, s.velocity_history[:, 0, 1], )


# noinspection PyUnresolvedReferences
@pytest.mark.parametrize("E0", np.linspace(-10, 10, 10))
def test_relativistic_harmonic_oscillator(_rela_pusher, _N_particles, E0):
    E0 = 1
    T = 10
    omega = 2 * np.pi / T
    dt = T / 200
    NT = helper_functions.calculate_number_timesteps(T, dt)
    g = Frame(dt, 1, 1, NT)
    s = Species(1, 1, _N_particles, g, pusher=_rela_pusher)
    t = np.arange(0, T, dt * s.save_every_n_iterations) - dt / 2

    t_s = t - dt / 2
    v_analytical = E0 * s.c * s.q * np.sin(omega * t_s) / np.sqrt(
        (E0 * s.q * np.sin(omega * t_s)) ** 2 + (s.m * omega * s.c) ** 2)

    def electric_field(x, t):
        return np.array([[1, 0, 0]], dtype=float) * E0 * np.cos(omega * t)

    s.init_push(lambda x: electric_field(x, 0))
    for i in range(NT):
        s.save_particle_values(i)
        s.push(lambda x: electric_field(x, i * dt))

    assert (s.velocity_history < 1).all(), plot(_rela_pusher, t, v_analytical, s.velocity_history[:, 0, 0],
                                                f"Velocity went over c! Max velocity: {s.velocity_history.max()}")
    assert np.allclose(s.velocity_history[:, 0, 0], v_analytical, atol=atol, rtol=rtol), \
        plot(_rela_pusher, t, v_analytical, s.velocity_history[:, 0, 0], )
