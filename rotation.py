"""Boris pusher, relativistic as hell, 1 particle"""
from Species import Species
import numpy as np
import matplotlib.pyplot as plt

q = 1
m = 1
dt = 0.1
c = 1
def rotation_matrix(t, s):
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

N = 10
x = np.ones(N)
v = np.array(N*[[1., 0, 0]])
# E = np.array([1., 2., 3.])
E = np.array(N*[[0., 0., 0.1]])
# B = np.array([1., 2., 3.])
B = np.array(N*[[0., 0., 1.]])


NT = 100
x_history = np.zeros((NT, N))
v_history = np.zeros((NT, N, 3))
time = np.arange(NT) * dt
for i in range(NT):
    x_history[i] = x
    v_history[i] = v
    vminus = v + q * E / m * dt * 0.5
    gamma_middle = np.sqrt(1+((vminus/c)**2).sum(axis=1, keepdims=True))

    # rotate to add magnetic field
    t = B * q * dt / (2 * m * gamma_middle)
    s = 2*t/(1+t*t)

    rot = rotation_matrix(t, s)

    # vplus = np.dot()
    # for matrix, vector in zip(rot, vminus):
    vplus = np.dot(rot, vminus)
    import ipdb; ipdb.set_trace()

    v_new = vplus + q * E / m * dt * 0.5
    gamma_new = np.sqrt(1+((vminus/c)**2).sum(axis=1, keepdims=True))
    x_new = x + v_new[0] / gamma_new * dt
    x, v = x_new, v_new

# print(x_history.shape)
plt.plot(time, x_history)
plt.plot(time, v_history)
plt.figure()
plt.plot(x_history, v_history[:,0])
plt.plot(v_history[:,0], v_history[:,1])
plt.show()
