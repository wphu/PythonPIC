# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import pyplot as plt

from ..algorithms.field_interpolation import charge_density_deposition
from ..classes import Particle, Species, TimelessGrid

from ..algorithms.field_interpolation import longitudinal_current_deposition, transversal_current_deposition


@pytest.fixture(params=np.arange(3, 4, 0.2))
def _position(request):
    return request.param


@pytest.fixture(params=np.arange(-0.9, 1, 0.2))
def _velocity(request):
    return request.param


def test_single_particle_longitudinal_deposition(_position, _velocity):
    g = TimelessGrid(L=7, NG=7)
    s = Particle(g, _position * g.dx, _velocity)
    dt = g.dx / s.c
    g.current_density[...] = 0
    # current_deposition(g, s, dt)
    longitudinal_current_deposition(g.current_density[:, 0], s.v[:, 0], s.x, dt * np.ones_like(s.x), g.dx, dt, s.q)
    collected_longitudinal_weights = g.current_density[:, 0].sum() / s.v[0, 0]

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
        ax.plot(g.x, g.current_density[1:-1, 0], "go-", alpha=0.7, linewidth=3, label=f"jx")
        ax.legend()
        plt.show()
        return title + " instead of 1"

    assert np.isclose(collected_longitudinal_weights, 1), plot_longitudinal()


def test_single_particle_transversal_deposition(_position, _velocity):
    g = TimelessGrid(L=7, NG=7)
    s = Particle(g, _position * g.dx, _velocity, 1, -1)
    dt = g.dx / s.c
    new_positions = s.x + s.v[:, 0] * dt
    g.current_density[...] = 0
    print("\n\n=====Start run===")
    # current_deposition(g, s, dt)
    transversal_current_deposition(g.current_density[:, 1:], s.v, s.x, dt * np.ones_like(s.x), g.dx, dt, s.q)
    total_currents = g.current_density[:, 1:].sum(axis=0) / s.v[0, 1:]
    total_sum_currents = g.current_density[:, 1:].sum()
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
            ax.plot(g.x + 0.5, g.current_density[1:-1, i], "o-", alpha=0.7, linewidth=i + 3, label=f"j{label}")
        ax.legend()
        plt.show()
        return title + " instead of 1"

    assert np.allclose(collected_transversal_weights, 1), plot_transversal()
    assert np.isclose(total_sum_currents, 0), (plot_transversal(), f"Currents do not zero out at {total_sum_currents}")


def test_two_particles_deposition(_position, _velocity):
    NG = 7
    L = NG
    g = TimelessGrid(L=L, NG=NG)
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
        longitudinal_current_deposition(g.current_density[:, 0], s.v[:, 0], s.x, dt * np.ones_like(s.x), g.dx, dt, s.q)
        transversal_current_deposition(g.current_density[:, 1:], s.v, s.x, dt * np.ones_like(s.x), g.dx, dt, s.q)
        # print(g.current_density)

    collected_weights = g.current_density.sum(axis=0) / np.array([_velocity, 1, -1], dtype=float)

    g2 = TimelessGrid(L=L, NG=NG)
    s = Species(1, 1, 2, g2)
    s.x[:] = positions
    s.v[:, 0] = _velocity
    s.v[:, 1] = 1
    s.v[:, 2] = -1
    # print("\n\n======TWO PARTICLES=======")
    # print(s)
    # print(s.x)
    # print(s.v)
    longitudinal_current_deposition(g2.current_density[:, 0], s.v[:, 0], s.x, dt * np.ones_like(s.x), g2.dx, dt, s.q)
    transversal_current_deposition(g2.current_density[:, 1:], s.v, s.x, dt * np.ones_like(s.x), g2.dx, dt, s.q)
    # print(g2.current_density)
    collected_weights2 = g2.current_density.sum(axis=0) / s.v[0, :]
    label = {0: 'x', 1: 'y', 2: 'z'}

    def plot():
        fig, (ax1, ax2) = plt.subplots(2)
        plt.suptitle(f"x: {positions}, vx: {_velocity}")
        for i in range(3):
            ax1.plot(g.x, g.current_density[1:-1, i], alpha=(3 - i) / 3, lw=1 + i, label=f"1 {label[i]}")
            ax1.plot(g.x, g2.current_density[1:-1, i], alpha=(3 - i) / 3, lw=1 + i, label=f"2 {label[i]}")
            ax2.plot(g.x, (g.current_density - g2.current_density)[1:-1, i], alpha=(3 - i) / 3, lw=1 + i,
                     label=f"1 - 2 {label[i]}")
        for ax in [ax1, ax2]:
            ax.scatter(s.x, np.zeros_like(s.x))
            ax.legend(loc='lower right')
            ax.set_xticks(g.x)
            ax.grid()
        fig.savefig(f"data_analysis/deposition/{_position:.2f}_{_velocity:.2f}.png")

    assert np.allclose(g.current_density, g2.current_density), ("Currents don't match!", plot())
    assert np.allclose(collected_weights, collected_weights2), "Weights don't match!"


@pytest.mark.parametrize("N", [100, 1000, 10000])
def test_many_particles_deposition(N, _velocity):
    NG = 10
    L = NG
    g = TimelessGrid(L=L, NG=NG)
    s = Species(1.0 / N, 1, N, g)
    s.distribute_uniformly(L, 1e-6, 2 * g.dx, 2 * g.dx)
    s.v[:, 0] = _velocity
    s.v[:, 1] = 1
    s.v[:, 2] = -1
    dt = g.dx / s.c
    g.gather_current([s])
    collected_weights = g.current_density.sum(axis=0) / s.v[0, :]
    label = {0: 'x', 1: 'y', 2: 'z'}

    def plot():
        fig, ax = plt.subplots()
        for i in range(3):
            ax.plot(g.x, g.current_density[1:-1, i], alpha=(3 - i) / 3, lw=1 + i, label=f"{label[i]}")
            ax.scatter(s.x, np.zeros_like(s.x))
            ax.legend(loc='lower right')
            ax.set_xticks(g.x)
            ax.grid()
        fig.savefig(f"data_analysis/deposition/multiple_{N}_{_velocity:.2f}.png")

    assert np.allclose(collected_weights, 1), ("Weights don't match!", plot())


if __name__ == '__main__':
    test_single_particle_transversal_deposition(3.01, 1)

    for position in (3.01, 3.25, 3.49, 3.51, 3.99):
        for velocity in (0.01, 1, -0.01, -1):
            test_single_particle_transversal_deposition(position, velocity)
