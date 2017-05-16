# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pytest

import algorithms_pusher
import helper_functions
from Species import Species
from helper_functions import l2_test

atol = 1e-1
rtol = 1e-4

@pytest.fixture(params=[1, 2, 3, 10, 100])
def _N(request):
    return request.param

@pytest.fixture(params=[algorithms_pusher.leapfrog_push, algorithms_pusher.boris_push])
def _pusher(request):
    return request.param

@pytest.fixture(params=[algorithms_pusher.rela_boris_push_lpic, algorithms_pusher.rela_boris_push_bl])
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
    plt.show()
    return message


def test_constant_field(_pusher, _N):
    T = 10
    dt = 10 / 200
    NT = helper_functions.calculate_NT(T, dt)
    s = Species(1, 1, _N, pusher=_pusher, NT=NT)
    t = np.arange(0, T, dt * s.save_every_n_iterations) - dt / 2

    def uniform_field(x):
        return np.array([[1, 0, 0]], dtype=float)

    x_analytical = 0.5 * (t+dt/2) ** 2 + 0
    s.init_push(uniform_field, dt)
    for i in range(NT):
        s.save_particle_values(i)
        s.push(uniform_field, dt)

    assert np.allclose(s.position_history[:, 0], x_analytical, atol=atol, rtol=rtol), \
        plot(_rela_pusher, t, x_analytical, s.position_history[:, 0])



def test_relativistic_constant_field(_rela_pusher, _N):
    T = 10
    dt = 10 / 200
    NT = helper_functions.calculate_NT(T, dt)
    s = Species(1, 1, _N, pusher=_rela_pusher, NT=NT)
    t = np.arange(0, T, dt * s.save_every_n_iterations) - dt / 2

    def uniform_field(x):
        return np.array([[1, 0, 0]], dtype=float)

    v_analytical = (t - dt / 2) / np.sqrt((t - dt / 2) ** 2 + 1)
    s.init_push(uniform_field, dt)
    for i in range(NT):
        s.save_particle_values(i)
        s.push(uniform_field, dt)

    assert (s.velocity_history < 1).all(), plot(_rela_pusher, t, v_analytical, s.velocity_history[:, 0, 0],
                                                f"Velocity went over c! Max velocity: {s.velocity_history.max()}")
    assert np.allclose(s.velocity_history[:, 0, 0], v_analytical, atol=atol, rtol=rtol), \
        plot(_rela_pusher, t, v_analytical, s.velocity_history[:, 0, 0], )

def test_relativistic_magnetic_field(_rela_pusher, _N, _v0):
    B0 = 1
    T = 10
    dt = T / 200
    NT = helper_functions.calculate_NT(T, dt)
    s = Species(1, 1, _N, pusher=_rela_pusher, NT=NT)
    t = np.arange(0, T, dt * s.save_every_n_iterations) - dt / 2
    print(s.v, _v0)
    s.v[:,1] = _v0

    def uniform_magnetic_field(x):
        return np.array([[0, 0, B0]], dtype=float)

    def uniform_electric_field(x):
        return np.zeros(3, dtype=float)

    gamma = algorithms_pusher.gamma_from_v(s.v, s.c)[0]
    vy_analytical = _v0 * np.cos(s.q * B0 * (t - dt/2) / (s.m * gamma))

    s.init_push(uniform_electric_field, dt, uniform_magnetic_field)
    for i in range(NT):
        s.save_particle_values(i)
        s.push(uniform_electric_field, dt, uniform_magnetic_field)
    assert (s.velocity_history < 1).all(), plot(_rela_pusher, t, vy_analytical, s.velocity_history[:, 0, 1],
                                                f"Velocity went over c! Max velocity: {s.velocity_history.max()}")
    assert np.allclose(s.velocity_history[:, 0, 1], vy_analytical, atol=atol, rtol=rtol),\
        plot(_rela_pusher, t, vy_analytical, s.velocity_history[:, 0, 1], )



# @pytest.mark.parametrize(["E0"], [[1]])
# def test_x2(E0):
#    def Efield(x):
#        return -E0 * (x-L/2)
#    def Bfield(x):
#        return B0*np.array((0,0,1))

# def test_boris_pusher():
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#     particles = Species(1, 1, 4, "particle")
#     particles.x = np.array([5, 5, 5, 5], dtype=float)
#     particles.v = np.zeros((particles.N,3),dtype=float)
#     NT = 1000
#     x_history = np.zeros((NT, particles.N))
#     v_history = np.zeros((NT, particles.N, 3))
#     T, dt = np.linspace(0, 2*np.pi*10, NT, retstep=True)
#     electric_field = lambda x: 1+np.arange(particles.N)
#     magnetic_field = lambda x: np.array(particles.N*[[0,0,1]])
#     particles.boris_init(electric_field, magnetic_field, dt, np.inf)
#     for i, t in enumerate(T):
#         x_history[i] = particles.x
#         v_history[i] = particles.v[:,:]
#         particles.boris_push_particles(electric_field, magnetic_field, dt, np.inf)
#
#     def get_frequency(x_history):
#         x_fft = (np.abs(np.fft.rfft(x_history, axis=0)))**2
#         x_fft[0,:] = 0
#         fft_freq = np.fft.rfftfreq(NT) * NT / 10
#         #TODO: how to renormalize this to take just one period
#         indices = np.argmax(x_fft, axis=0) #this here thing
#         # plt.plot(fft_freq, x_fft, ".-")
#         # for a in [fft_freq[indices], ]:
#         #     print(indices, a.shape, a)
#         # plt.scatter(fft_freq[indices], x_fft[indices, np.arange(4)])
#         # plt.show()
#         return fft_freq[indices]
#
#     cyclotron_frequency = particles.q/particles.m * 1
#     print(cyclotron_frequency)
#     print(get_frequency(x_history))
#     def plot():
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         for i in range(4):
#             ax.plot(T, x_history[:,i], v_history[:,i,0], "-")
#         plt.show()
#     assert np.isclose(cyclotron_frequency, get_frequency(x_history)).all(), plot()


# def test_ramp_field():
#     s = Species(1, 1, 1)
#     s.x = np.array([0.25], dtype=float)
#     print(s.x)
#     t, dt = np.linspace(0, 1, 200, retstep=True, endpoint=False)
#     ramp_field = lambda x: x - 0.5
#     x_analytical = np.exp(t)
#     x_data = []
#     s.init_push(ramp_field, dt)
#     for i in range(t.size):
#         x_data.append(s.x[0])
#         s.push(ramp_field, dt, np.inf)
#     x_data = np.array(x_data)
#     print(x_analytical - x_data)
#
#     def plot():
#         fig, (ax1, ax2) = plt.subplots(2, sharex=True)
#         ax1.plot(t, x_analytical, "b-", label="analytical result")
#         ax1.plot(t, x_data, "ro--", label="simulation result")
#         ax1.legend()
#
#         ax2.plot(t, x_data - x_analytical, label="difference")
#         ax2.legend()
#
#         ax2.set_xlabel("t")
#         ax1.set_ylabel("x")
#         ax2.set_ylabel("delta x")
#         plt.show()
#         return "ramp field is off"
#
#     assert l2_test(x_analytical, x_data), plot()

if __name__ == "__main__":
    test_relativistic_constant_field(algorithms_pusher.rela_boris_push_lpic)
