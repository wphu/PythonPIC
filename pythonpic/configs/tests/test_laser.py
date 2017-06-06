# coding=utf-8
import pytest
import numpy as np
import matplotlib.pyplot as plt
from pythonpic.helper_functions.helpers import make_sure_path_exists
from pythonpic.visualization.time_snapshots import SpatialPerturbationDistributionPlot
from pythonpic.classes import Particle

from . import on_failure
from ..run_laser import laser, lightspeed, npic, electric_charge, electron_rest_mass

# @pytest.mark.parametrize("std", [0])
# def test_stability(std):
#     S = laser("stability_test", 1000, 0, 0, std).test_run()
#     def plots():
#         fig, ax = plt.subplots()
#         fig.suptitle(std)
#         plot = SpatialPerturbationDistributionPlot(S, ax)
#         plot.update(S.NT-1)
#         make_sure_path_exists(S.filename)
#         fig.savefig(S.filename.replace(".hdf5", ".png"))
#         plt.close(fig)
#     s = S.list_species[0]
#     assert np.allclose(s.density_history[0], s.density_history[-1]), plots()

def test_longitudinal_current():
    S = laser("test_current", 0, 0, 0, 0)
    print(f"dx: {S.grid.dx}, dt: {S.grid.dt}, Neuler: {S.grid.NG}")
    p = Particle(S.grid,
                 9.45*S.grid.dx,
                 0.9*lightspeed,
                 q=-electric_charge,
                 m=electron_rest_mass,
                 scaling=npic)
    print(p.q, p.m, p.scaling)
    S.grid.list_species = [p]
    S.grid.gather_current([p])
    investigated_density = S.grid.current_density_x[10:12]
    target_density = np.array([-2.3838e13, -4.0184e14])
    error = (investigated_density - target_density) /target_density * 100
    for a, b, c, d in zip(np.arange(10, 12)-1, investigated_density, target_density, error):
        print(f"{a} {b:.9e} {c:.9e} E{d:.3f}%")
    assert np.allclose(investigated_density, target_density, rtol=1e-2)

# def test_transversal_current():
#     S = laser("test_current", 0, 0, 0, 0)
#     p = Particle(S.grid,
#                  9.95*S.grid.dx,
#                  0.9*lightspeed,
#                  0.01*lightspeed,
#                  q=-electric_charge,
#                  m=electron_rest_mass,
#                  scaling=npic)
#     S.grid.list_species = [p]
#     S.grid.gather_current([p])
#     for a, b in zip(np.arange(20)-2, S.grid.current_density_yz[:20, 0]):
#         print(f"{a} {b:.9e}")
#     assert False
#
# def test_pusher():
#     S = laser("test_current", 0, 0, 0, 0)
#     p = Particle(S.grid,
#                  9.45*S.grid.dx,
#                  0,
#                  0,
#                  q=-electric_charge,
#                  m=electron_rest_mass,
#                  scaling=npic)
#     S.grid.list_species = [p]
#     E = lambda x: np.array([[0, 1.228e12, 0]])
#     B = lambda x: np.array([[0,0,4.027e3]])
#     p.push(E, B)
#     print(p.v)
#     a, b, c = p.v[0]
#     print(f"vx:{a:.9e}\n"
#           f"vy:{b:.9e}\n"
#           f"vz:{c:.9e}")
#     assert False
