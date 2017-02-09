"""Boris pusher, relativistic as hell, 1 particle"""
from Species import Species
import numpy as np
import matplotlib.pyplot as plt
# import numba
import time

# @numba.njit() #TODO: add numba to this algorithm
def rotation_matrix(t, s, N):
    result = np.zeros((N, 3, 3))
    result[:] = np.eye(3)

    sz = s[:,2]
    sy = s[:,1]
    tz = t[:,2]
    ty = t[:,1]
    sztz = sz * tz
    syty = sy * ty
    result[:,0,0] -= sztz
    result[:,0,0] -= syty
    result[:,0,1] = sz
    result[:,1,0] = -sz
    result[:,0,2] = -sy
    result[:,2,0] = sy
    result[:,1,1] -= sztz
    result[:,2,2] -= syty
    result[:,2,1] = sy*tz
    result[:,1,2] = sz*ty
    return result

# @numba.njit()
def rela_boris_push(x, v, E, B, q, m, dt, c=1):
    """
    relativistic Boris pusher
    """
    vminus = v + q * E / m * dt * 0.5
    N = x.size
    gamma_middle = np.sqrt(1+((vminus/c)**2).sum(axis=1, keepdims=True))

    # rotate to add magnetic field
    t = B * q * dt / (2 * m * gamma_middle)
    s = 2*t/(1+t*t)

    rot = rotation_matrix(t, s, N)

    vplus = np.einsum('ijk,ik->ij',rot,vminus)
    # import ipdb; ipdb.set_trace()

    v_new = vplus + q * E / m * dt * 0.5
    gamma_new = np.sqrt(1+((vminus/c)**2).sum(axis=1))
    # import ipdb; ipdb.set_trace()

    x_new = x + v_new[:,0] / gamma_new * dt
    return  x_new, v_new

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

if __name__=="__main__":
    test_rela_boris()
