# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pytest

from Grid import Grid
from Species import Particle
from algorithms_interpolate import longitudinal_current_deposition, transversal_current_deposition


@pytest.fixture(params=(3.01, 3.25, 3.49, 3.51, 3.99))
def _position(request):
    return request.param


@pytest.fixture(params=(0.01, 1, -0.01, -1))
def _velocity(request):
    return request.param


def test_single_particle_longitudinal_deposition(_position, _velocity):
    g = Grid(NG=7, L=7)
    s = Particle(_position * g.dx, _velocity)
    dt = s.c * g.dx
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
    g = Grid(NG=7, L=7)
    s = Particle(_position * g.dx, _velocity, 1, -1)
    dt = s.c * g.dx
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

    assert np.isclose(collected_transversal_weights, 1).all(), plot_transversal()
    assert np.isclose(total_sum_currents, 0), (plot_transversal(), f"Currents do not zero out at {total_sum_currents}")


if __name__ == '__main__':
    test_single_particle_transversal_deposition(3.01, 1)

    for position in (3.01, 3.25, 3.49, 3.51, 3.99):
        for velocity in (0.01, 1, -0.01, -1):
            test_single_particle_transversal_deposition(position, velocity)
