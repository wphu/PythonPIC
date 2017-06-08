# coding=utf-8
import numpy as np

from pythonpic.classes import Particle
from pythonpic.configs.run_laser import laser, electric_charge, electron_rest_mass, npic


def test_pusher():
    S = laser("test_current", 0, 0, 0, 0)
    p = Particle(S.grid,
                 9.45*S.grid.dx,
                 0,
                 0,
                 q=-electric_charge,
                 m=electron_rest_mass,
                 scaling=npic)
    S.grid.list_species = [p]
    Ev, Bv = (np.array([[0, 1.228e12, 0]]), np.array([[0,0,4.027e3]]))
    drift_vel = np.cross(Ev, Bv)/(Bv.sum()**2)
    print(drift_vel)
    E = lambda x: (Ev, Bv)
    p.push(E)
    print(p.v)
    a, b, c = p.v[0]
    e = p.energy
    print(f"vx:{a:.9e}\n"
          f"vy:{b:.9e}\n"
          f"vz:{c:.9e}\n"
          f"KE:{e:.9e}\n")
    assert False
