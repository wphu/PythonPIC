# coding=utf-8
import pytest
import numpy as np
import pandas as pd
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

@pytest.mark.parametrize(["init_pos", "init_vx", "expected"], [
    [9.45, 0.9, np.array([0, 0.056, 0.944, 0])], # cases 3, 4
    [9.55, 0.9, np.array([0, 0, 1, 0])], # cases 3, 4
    [9.95, 0.9, np.array([0, 0, 0.611, 0.389])], # cases 3, 4
    [9.05, 0.9, np.array([0, 0.5, 0.5, 0])], # cases 3, 4
    [9.45, -0.9, np.array([0, 1, 0, 0])], # cases 1, 2
    [9.55, -0.9, np.array([0, 0.944, 0.056, 0])], # cases 1, 2
    [9.95, -0.9, np.array([0, 0.5, 0.5, 0])], # cases 1, 2
    [9.05, -0.9, np.array([0.389, 0.611, 0, 0])], # cases 1, 2

    [9.05, 0.0, np.array([0, 0, 0, 0])],
    [9.05, 0.1, np.array([0, 1, 0, 0])], # cases 3, 4
    [9.45, 0.1, np.array([0, 0.5, 0.5, 0])], # cases 3, 4
    [9.55, 0.1, np.array([0, 0, 1, 0])], # cases 3, 4
    [9.95, 0.1, np.array([0, 0, 1, 0])], # cases 3, 4
    [9.05, -0.1, np.array([0, 1, 0, 0])], # cases 3, 4
    [9.45, -0.1, np.array([0, 1, 0, 0])], # cases 3, 4
    [9.55, -0.1, np.array([0, 0.5, 0.5, 0])], # cases 3, 4
    [9.95, -0.1, np.array([0, 0, 1, 0])], # cases 3, 4
    ])
def test_longitudinal_current(init_pos, init_vx, expected):
    S = laser("test_current", 0, 0, 0, 0)
    print(f"dx: {S.grid.dx}, dt: {S.grid.dt}, Neuler: {S.grid.NG}")
    p = Particle(S.grid,
                 init_pos*S.grid.dx,
                 init_vx*lightspeed,
                 q=-electric_charge,
                 m=electron_rest_mass,
                 scaling=npic)
    S.grid.list_species = [p]
    S.grid.gather_current([p])
    investigated_density = S.grid.current_density_x[9:13] /(p.eff_q * init_vx * lightspeed)
    if init_vx == 0.0:
        investigated_density[...] = 0

    target_density = expected
    error = (investigated_density - target_density) /target_density * 100
    error[(investigated_density - target_density) == 0] = 0
    print(pd.DataFrame({"indices": np.arange(9, 13)-1,
                        "found density":investigated_density,
                        "target density":target_density,
                        "error %":error}))
    assert np.allclose(target_density, investigated_density, rtol=1e-2, atol = 1e-3)

@pytest.mark.parametrize(["init_pos", "init_vx", "expected"], [
    [9.45, 0.9, np.array([0, 0.001, 0.597, 0.401, 0])], # X c 1 4 2
    [9.45, 0.1, np.array([0, 0.012, 0.975, 0.013, 0])], # X c 1 3
    [9.45, -0.1, np.array([0, 0.1, 0.9, 0, 0])], # c 2
    [9.45, -0.9, np.array([0, 0.5, 0.5, 0, 0])], # c 1 3
    [9.55, -0.9, np.array([0, 0.401, 0.597, 0.001, 0])], # X c 4 1 3
    [9.55, -0.1, np.array([0, 0.013, 0.975, 0.012, 0])], # X c
    [9.55, 0.1, np.array([0, 0, 0.9, 0.1, 0])], # c 3
    [9.55, 0.9, np.array([0, 0, 0.5, 0.5, 0])], # c 4 2
    [9.05, -0.9, np.array([0.068, 0.764, 0.168, 0, 0])], # c 1 4 2
    [9.05, -0.1, np.array([0, 0.5, 0.5, 0, 0])], # c 1 3
    [9.05, 0.1, np.array([0, 0.4, 0.6, 0, 0])], # c 2
    [9.05, 0.9, np.array([0, 0.112, 0.775, 0.113, 0])], # c 1 3
    [9.95, -0.9, np.array([0, 0.113, 0.775, 0.112, 0])], # c
    [9.95, -0.1, np.array([0, 0, 0.6, 0.4, 0])], # c
    [9.95, 0.1, np.array([0, 0, 0.5, 0.5, 0])], # c
    [9.95, 0.9, np.array([0, 0, 0.168, 0.764, 0.068])], # c
    [9.95, 0, np.array([0, 0, 0.550, 0.450, 0])], # c
    [9.55, 0, np.array([0, 0, 0.950, 0.050, 0])], # c
    [9.45, 0, np.array([0, 0.050, 0.950, 0, 0])], # c
    [9.05, 0, np.array([0, 0.450, 0.550, 0, 0])], # c
    [9.5, 0, np.array([0, 0, 1, 0, 0])], # c
    [9.5, -0.9, np.array([0, 0.450, 0.550, 0, 0])], # c
    [9.5, -0.1, np.array([0, 0.05, 0.95, 0, 0])], # c
    [9.5, 0.1, np.array([0, 0, 0.950, 0.05, 0])], # c
    [9.5, 0.9, np.array([0, 0, 0.550, 0.450, 0])], # c
    ])
def test_transversal_current(init_pos, init_vx, expected):
    S = laser("test_current", 0, 0, 0, 0)
    print(f"dx: {S.grid.dx}, dt: {S.grid.dt}, Neuler: {S.grid.NG}")
    init_vy = 0.01
    p = Particle(S.grid,
                 init_pos*S.grid.dx,
                 init_vx*lightspeed,
                 init_vy*lightspeed,
                 q=-electric_charge,
                 m=electron_rest_mass,
                 scaling=npic)
    S.grid.list_species = [p]
    S.grid.gather_current([p])
    investigated_density = S.grid.current_density_yz[9:14, 0] / p.eff_q / init_vy / lightspeed
    target_density = expected
    error = (investigated_density - target_density) * 100
    error[investigated_density != 0] /= investigated_density[investigated_density !=0]
    print(pd.DataFrame({"indices": np.arange(9, 14)-2,
                       "found density":investigated_density,
                       "target density":target_density,
                       "error %":error}))
    assert np.allclose(investigated_density, target_density, rtol=1e-2, atol=1e-3)
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
