import numpy as np
from Species import Species, RelativisticSpecies
import numpy as np
import matplotlib.pyplot as plt
# import numba
import time
from Grid import RelativisticGrid


def test_rela_boris():
    q = 1
    m = 1
    dt = 0.001
    c = 1
    N = 100
    x = np.zeros(N)
    v = np.linspace(0,0.99*c, N)[:, np.newaxis]*np.array([[1., 0, 0]])
    v[:,0] += np.random.normal(size=N, scale=1e-2)
    v[:,0] %= c
    # E = np.array([1., 2., 3.])
    E = np.array(N*[[0., 0., 0]])
    # B = np.array([1., 2., 3.])
    B = np.array(N*[[0., 0., 1.]])

    NT = 100000
    x_history = np.zeros((NT, N))
    v_history = np.zeros((NT, N, 3))
    t = np.arange(NT) * dt
    start_time = time.time()
    for i in range(NT):
        x_history[i] = x
        v_history[i] = v
        x, v = rela_boris_push(x, v, E, B, q, m, dt, c)
    runtime = time.time() - start_time
    print(f"Runtime was {runtime:.3f} s")

    plt.plot(t, x_history)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.figure()
    plt.plot(t, v_history[:,:,0], label="vx")
    plt.plot(t, v_history[:,:,1], label="vy")
    plt.plot(t, v_history[:,:,2], label="vz")
    plt.xlabel("t")
    plt.ylabel("v")
    plt.legend()
    plt.figure()
    plt.plot(x_history, v_history[:,:,0])
    plt.xlabel("x")
    plt.ylabel("v")
    plt.show()

# def test_rela_boris2():
#     N=100
#     NT = 100000
#     species = RelativisticSpecies(1, 1, N, NT)
#     c = 1
#     dt = 0.001
#     species.v = np.linspace(0,0.99*c, N)[:, np.newaxis]*np.array([[1., 0, 0]])
#     species.v[:,0] += np.random.normal(size=N, scale=1e-2)
#     species.v[:,0] %= c
#
#     E = np.array(N*[[0., 0., 0]])
#     B = np.array(N*[[0., 0., 1.]])

if __name__=="__main__":
    test_rela_boris()
