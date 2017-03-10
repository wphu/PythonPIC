from Species import Species
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import l2_test


def test_constant_field(plotting=False):
    s = Species(1, 1, 1)
    # s.x = np.array([0], dtype=float)
    print(s.x)
    def uniform_field(x):
        return np.ones_like(x)
    t, dt = np.linspace(0, 10, 200, retstep=True, endpoint=False)
    x_analytical = 0.5 * t**2 + 0
    x_data = []
    for i in range(t.size):
        x_data.append(s.x[0])
        s.push(uniform_field, dt, np.inf)
    x_data = np.array(x_data)
    print(x_analytical - x_data)

    def plot():
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(t, x_analytical, "b-", label="analytical result")
        ax1.plot(t, x_data, "ro--", label="simulation result")
        ax1.legend()

        ax2.plot(t, x_data - x_analytical, label="difference")
        ax2.legend()

        ax2.set_xlabel("t")
        ax1.set_ylabel("x")
        ax2.set_ylabel("delta x")
        plt.show()
        return None
    if plotting:
        plot()

    assert l2_test(x_analytical, x_data), plot()

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
#     for i in range(t.size):
#         x_data.append(s.x[0])
#         s.leapfrog_push(ramp_field, dt, np.inf)
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
    test_constant_field()
