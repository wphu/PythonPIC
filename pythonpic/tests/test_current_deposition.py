# coding=utf-8
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from pythonpic.configs.run_laser import laser, npic
from pythonpic.helper_functions.physics import lightspeed, electric_charge, electron_rest_mass

from pythonpic.helper_functions.helpers import make_sure_path_exists
from pythonpic.visualization.plotting import plots
from pythonpic.visualization import animation
from pythonpic.algorithms.current_deposition import longitudinal_current_deposition, transversal_current_deposition
from pythonpic.classes import Particle, Species, Grid, Simulation


@pytest.fixture(params=np.arange(3, 4, 0.2))
def _position(request):
    return request.param


@pytest.fixture(params=np.arange(-0.9, 1, 0.2))
def _velocity(request):
    return request.param


@pytest.fixture(params=(True, False))
def _truefalse(request):
    return request.param

_truefalse2 = _truefalse

def test_single_particle_longitudinal_deposition(_position, _velocity):
    g = Grid(T = 1, L=7, NG=7)
    s = Particle(g, _position * g.dx, _velocity)
    dt = g.dt
    g.current_density_x[...] = 0
    g.current_density_yz[...] = 0
    # current_deposition(g, s, dt)
    longitudinal_current_deposition(g.current_density_x, s.v[:, 0], s.x, g.dx, dt, s.q)
    collected_longitudinal_weights = g.current_density_x.sum() / s.v[0, 0]

    def plot_longitudinal():
        fig, ax = plt.subplots()
        ax.scatter(s.x, 0)
        new_positions = s.x + s.v[:, 0] * dt
        title = f"x0: {s.x[0]} v: {s.v[0,0]} x1: {new_positions[0]} w: {collected_longitudinal_weights}"
        ax.annotate(title, xy=(new_positions[0], 0), xytext=(s.x[0], 0.3),
                    arrowprops=dict(facecolor='black', shrink=0.05, linewidth=0.5), horizontalalignment='right')
        ax.set_xticks(g.x)
        ax.set_xticks(np.arange(0, g.L, g.dx / 2), minor=True)
        ax.grid(which='minor', alpha=0.3)
        ax.grid(which='major', alpha=0.7)
        ax.set_title(title)
        ax.scatter(new_positions, 0)
        ax.plot(g.x, g.current_density_x[1:-2], "go-", alpha=0.7, linewidth=3, label=f"jx")
        ax.legend()
        plt.show()
        return title + " instead of 1"

    assert np.isclose(collected_longitudinal_weights, 1), plot_longitudinal()


def test_single_particle_transversal_deposition(_position, _velocity):
    g = Grid(1,L=7, NG=7)
    s = Particle(g, _position * g.dx, _velocity, 1, -1)
    dt = g.dx / s.c
    new_positions = s.x + s.v[:, 0] * dt
    g.current_density_x[...] = 0
    g.current_density_yz[...] = 0
    print("\n\n=====Start run===")
    # current_deposition(g, s, dt)
    transversal_current_deposition(g.current_density_yz, s.v, s.x, g.dx, dt, s.q)
    total_currents = g.current_density_yz.sum(axis=0) / s.v[0, 1:]
    total_sum_currents = g.current_density_yz.sum()
    x_velocity = s.v[0, 0]
    collected_transversal_weights = total_currents
    print("total", total_currents)
    print("x velocity", x_velocity)
    print("weights", collected_transversal_weights)

    def plot_transversal():
        fig, ax = plt.subplots()
        ax.scatter(s.x, 0)
        title = f"x0: {s.x[0]} v: {s.v[0,0]} x1: {new_positions[0]:.3f} w: {collected_transversal_weights}"
        ax.annotate(title, xy=(new_positions[0], 0), xytext=(s.x[0], 0.3),
                    arrowprops=dict(facecolor='black', shrink=0.05, linewidth=0.5), horizontalalignment='right')
        ax.set_xticks(g.x)
        ax.set_xticks(np.arange(0, g.L, g.dx / 2), minor=True)
        ax.grid(which='minor', alpha=0.3)
        ax.grid(which='major', alpha=0.7)
        ax.set_title(title)
        ax.scatter(new_positions, 0)
        for i, label in {1: 'y', 2: 'z'}.items():
            ax.plot(g.x + 0.5, g.current_density_yz[1:-1, i - 1], "o-", alpha=0.7, linewidth=i + 3, label=f"j{label}")
        ax.legend()
        plt.show()
        return title + " instead of 1"

    assert np.allclose(collected_transversal_weights, 1), plot_transversal()
    assert np.isclose(total_sum_currents, 0), (plot_transversal(), f"Currents do not zero out at {total_sum_currents}")


def test_single_particle_above_lightspeed():
    g = Grid(1,L=7, NG=7)
    s = Particle(g, 1*g.dx, g.c*4, g.c*2)
    dt = g.dx / s.c
    g.current_density_x[...] = 0
    g.current_density_yz[...] = 0
    with pytest.raises(Exception):
        longitudinal_current_deposition(g.current_density_x, s.v[:, 0], s.x, g.dx, dt, s.q)
    with pytest.raises(Exception):
        transversal_current_deposition(g.current_density_yz, s.v, s.x, g.dx, dt, s.q)


def test_two_particles_deposition(_position, _velocity, _truefalse, _truefalse2):
    NG = 7
    L = NG
    g = Grid(1,L=L, NG=NG)
    c = 1
    dt = g.dx / c
    positions = [_position * g.dx, (L - _position * g.dx) % L]
    # print(positions)
    for position in positions:
        s = Particle(g, position, _velocity, 1, -1)
        # print(f"\n======PARTICLE AT {position}=======")
        # print(s)
        # print(s.x)
        # print(s.v)
        if _truefalse:
            longitudinal_current_deposition(g.current_density_x, s.v[:, 0], s.x, g.dx, dt, s.q)
        if _truefalse2:
            transversal_current_deposition(g.current_density_yz, s.v, s.x, g.dx, dt, s.q)
        # print(g.current_density)

    collected_weights_x = g.current_density_x.sum(axis=0) / _velocity
    collected_weights_yz = g.current_density_yz.sum(axis=0) / np.array([1, -1], dtype=float)

    g2 = Grid(1,L=L, NG=NG)
    s = Species(1, 1, 2, g2)
    s.x[:] = positions
    s.v[:, 0] = _velocity
    s.v[:, 1] = 1
    s.v[:, 2] = -1
    # print("\n\n======TWO PARTICLES=======")
    # print(s)
    # print(s.x)
    # print(s.v)
    if _truefalse:
        longitudinal_current_deposition(g2.current_density_x, s.v[:, 0], s.x, g2.dx, dt, s.q)
    if _truefalse2:
        transversal_current_deposition(g2.current_density_yz, s.v, s.x, g2.dx, dt, s.q)
    # print(g2.current_density)
    collected_weights_x2 = g2.current_density_x.sum(axis=0) / s.v[0, 0]
    collected_weights_yz2 = g2.current_density_yz.sum(axis=0) / np.array([1, -1], dtype=float)
    label = {0: 'x', 1: 'y', 2: 'z'}

    def plot():
        fig, (ax1, ax2) = plt.subplots(2)
        plt.suptitle(f"x: {positions}, vx: {_velocity}")
        i = 0
        ax1.plot(g.x, g.current_density_x[1:-2], alpha=(3 - i) / 3, lw=1 + i, label=f"1 {label[i]}")
        ax1.plot(g.x, g2.current_density_x[1:-2], alpha=(3 - i) / 3, lw=1 + i, label=f"2 {label[i]}")
        ax2.plot(g.x, (g.current_density_x - g2.current_density_x)[1:-2], alpha=(3 - i) / 3, lw=1 + i,
                 label=f"1 - 2 {label[i]}")
        for i in range(1, 3):
            ax1.plot(g.x, g.current_density_yz[2:-2, i-1], alpha=(3 - i) / 3, lw=1 + i, label=f"1 {label[i]}")
            ax1.plot(g.x, g2.current_density_yz[2:-2, i-1], alpha=(3 - i) / 3, lw=1 + i, label=f"2 {label[i]}")
            ax2.plot(g.x, (g.current_density_yz - g2.current_density_yz)[2:-2, i-1], alpha=(3 - i) / 3, lw=1 + i,
                     label=f"1 - 2 {label[i]}")
        for ax in [ax1, ax2]:
            ax.scatter(s.x, np.zeros_like(s.x))
            ax.legend(loc='lower right')
            ax.set_xticks(g.x)
            ax.grid()
        fig.savefig(f"data_analysis/deposition/{_position:.2f}_{_velocity:.2f}.png")

    assert np.allclose(g.current_density_x, g2.current_density_x), ("Longitudinal currents don't match!", plot())
    assert np.allclose(g.current_density_yz, g2.current_density_yz), ("Transversal currents don't match!", plot())
    assert np.allclose(collected_weights_x, collected_weights_x2), ("Longitudinal weights don't match!", plot())
    assert np.allclose(collected_weights_yz, collected_weights_yz2), ("Transversal weights don't match!", plot())


@pytest.mark.parametrize("N", [100, 1000, 10000])
def test_many_particles_deposition(N, _velocity):
    NG = 10
    L = NG
    g = Grid(1,L=L, NG=NG)
    s = Species(1.0 / N, 1, N, g)
    s.distribute_uniformly(L, 1e-6, 2 * g.dx, 2 * g.dx)
    s.v[:, 0] = _velocity
    s.v[:, 1] = 1
    s.v[:, 2] = -1
    dt = g.dx / s.c
    g.gather_current([s])
    longitudinal_collected_weights = g.current_density_x.sum(axis=0) / s.v[0, 0]
    transversal_collected_weights = g.current_density_yz.sum(axis=0) / s.v[0, 1:]
    label = {0: 'x', 1: 'y', 2: 'z'}

    def plot():
        fig, ax = plt.subplots()
        i = 0
        ax.plot(g.x, g.current_density_x[1:-2], alpha=(3 - i) / 3, lw=1 + i, label=f"{label[i]}")
        ax.scatter(s.x, np.zeros_like(s.x))
        ax.legend(loc='lower right')
        ax.set_xticks(g.x)
        ax.grid()
        for i in range(1,3):
            ax.plot(g.x, g.current_density_yz[2:-2, i-1], alpha=(3 - i) / 3, lw=1 + i, label=f"{label[i]}")
            ax.scatter(s.x, np.zeros_like(s.x))
            ax.legend(loc='lower right')
            ax.set_xticks(g.x)
            ax.grid()
        fig.savefig(f"data_analysis/deposition/multiple_{N}_{_velocity:.2f}.png")

    assert np.allclose(longitudinal_collected_weights, 1), ("Longitudinal weights don't match!", plot())
    assert np.allclose(transversal_collected_weights, 1), ("Transversal weights don't match!", plot())

@pytest.mark.parametrize("N", [100, 1000, 10000])
def test_many_particles_periodic_deposition(N, _velocity):
    NG = 10
    L = NG
    g = Grid(1,L=L, NG=NG, periodic=True)
    s = Species(1.0 / N, 1, N, g)
    s.distribute_uniformly(L)
    s.v[:, 0] = _velocity
    s.v[:, 1] = 1
    s.v[:, 2] = -1
    dt = g.dx / s.c
    g.gather_current([s])
    longitudinal_collected_weights = g.current_density_x[1:-2].sum(axis=0) / s.v[0, 0]
    transversal_collected_weights = g.current_density_yz[2:-2].sum(axis=0) / s.v[0, 1:]
    label = {0: 'x', 1: 'y', 2: 'z'}

    def plot():
        fig, ax = plt.subplots()
        i = 0
        fig.suptitle(f"Longitudinal weights: {longitudinal_collected_weights}, transversal: {transversal_collected_weights}")
        ax.plot(g.x, g.current_density_x[1:-2], alpha=(3 - i) / 3, lw=1 + i, label=f"{label[i]}")
        ax.scatter(s.x, np.zeros_like(s.x))
        ax.legend(loc='lower right')
        ax.set_xticks(g.x)
        ax.grid()
        for i in range(1,3):
            ax.plot(g.x, g.current_density_yz[2:-2, i-1], alpha=(3 - i) / 3, lw=1 + i, label=f"{label[i]}")
            ax.scatter(s.x, np.zeros_like(s.x))
            ax.legend(loc='lower right')
            ax.set_xticks(g.x)
            ax.grid()
        filename = f"data_analysis/test/periodic_multiple_{N}_{_velocity:.2f}.png"
        make_sure_path_exists(filename)
        fig.savefig(filename)
        fig.clf()
        plt.close(fig)

    assert np.allclose(longitudinal_collected_weights, 1), ("Longitudinal weights don't match!", plot())
    assert np.allclose(transversal_collected_weights, 1), ("Transversal weights don't match!", plot())


@pytest.mark.parametrize("periodic", [False, True])
def test_single_particle_deposition_simulation(periodic):
    g = Grid(T=100, L=1, NG=100, c=1, periodic=periodic)
    s = Particle(g, g.L / 2, g.dx, g.dx, -g.dx)
    sim = Simulation(g, [s], category_type="test",
                     filename=f"test_single_particle_depo_sim_{'' if periodic else 'a'}periodic")
    try:
        sim.run().postprocess()
    except Exception as E:
        assert False, (E, plots(sim, show_animation=True, show_static=True))

if __name__ == '__main__':
    test_single_particle_transversal_deposition(3.01, 1)

    for position in (3.01, 3.25, 3.49, 3.51, 3.99):
        for velocity in (0.01, 1, -0.01, -1):
            test_single_particle_transversal_deposition(position, velocity)


# @pytest.mark.parametrize(["T", "n_end_moat", "perturbation_amplitude",], [[5, 3, 0.3], [5, 5, 0.3],[5, 10, 0.3], [2.5, 20, 0.3]])
# def test_simulation_at_boundaries(T, n_end_moat, perturbation_amplitude):
#      g = Grid(T=T, L=1, NG=100, c=1, periodic=False)
#      s_n = Species(-1, 2000, 1000, g, "heavy electrons")
#      s = Species(+1, 1, 1000, g, "light protons")
#      s.v[:,0] = +0.3
#      s.distribute_uniformly(g.L, start_moat=g.dx*(g.NG-2-n_end_moat), end_moat=n_end_moat*g.dx)
#      s_n.distribute_uniformly(g.L, start_moat=g.dx*(g.NG-2-n_end_moat), end_moat=n_end_moat*g.dx)
#      filename = f"test_simulation_boundaries_{T}_{n_end_moat}"
#      sim = Simulation(g, [s, s_n], category_type="test", filename=filename)
#      sim.run().postprocess()
#      assert False, plots(sim, show_animation=True, animation_type=animation.OneDimAnimation, frames="all")
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