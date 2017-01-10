"""Boris pusher, relativistic as hell, 1 particle"""
from Species import Species

q = 1
m = 1
dt = 1

v = np.array([1., 2., 3.])
E = np.array([1., 2., 3.])
B = np.array([1., 2., 3.])

vminus = v + q * E / m * dt * 0.5
gamma = np.sqrt(1+((vminus/c)**2).sum())

# rotate to add magnetic field
t = B * q * dt / (2 * m * gamma)
s = 2*t/(1+t*t)

def rotation_matrix(t, s):
    result = np.eye(3)
    sz = s[2]
    sy = s[1]
    tz = t[2]
    ty = t[1]
    sztz = sz * tz
    syty = sy * ty
    result[0,0] -= sztz
    result[0,0] -= syty
    result[0,1] = sz
    result[1,0] = -sz
    result[0,2] = -sy
    result[2,0] = sy
    result[1,1] -= sztz
    result[2,2] -= syty
    result[2,1] = sy*tz
    result[1,2] = sz*ty
    return result

rot = rotation_matrix(t, s)
print(rot)


vprime = vminus + np.cross(vminus, t) # TODO: axis?
vplus = vminus + np.cross(vprime, s)
v_new = vplus + q * efield / m * dt * 0.5

energy = v * v_new * (0.5 * m)
v = v_new
return energy
