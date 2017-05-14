# coding=utf-8
import numpy as np
import pytest
import matplotlib.pyplot as plt

from Grid import Grid
from Species import Species, Particle
from algorithms_interpolate import longitudinal_current_deposition, transversal_current_deposition


@pytest.mark.parametrize("power", range(2, 8))
def test_longitudinal_deposition(power):
    N = 10 ** power
    s = Species(1 / N, 1, N, "test particles", 1, 1, 1)
    g = Grid()
    s.distribute_uniformly(g.L, 1e-6, g.L / 4, g.L / 4)
    # s.v[:, 0] = np.random.normal(scale=0.1, size=N)
    # s.v[:,0] = s.c*0.9
    s.v[:, 0] = g.dx / 100
    dt = g.dx

    longitudinal_current_deposition(g.current_density[:, 0], s.v[:, 0], s.x, dt, g.dx, dt, s.q)
    transversal_current_deposition(g.current_density[:, 1:], s.v, s.x, dt*np.ones_like(s.x), g.dx, dt, s.q)
    plt.plot(g.x, g.current_density[1:-1, 0], label=N)
    print(g.dx * g.current_density[:,0].sum())

    total_gathered_current = (g.current_density[1:-1, 0].sum() / (s.v[:, 0].mean()))
    assert np.isclose(total_gathered_current, 1, rtol=1e-1), "Longitudinal current deposition seems off!"
    expected_current = np.zeros_like(g.x)
    inside_moat = (g.x > g.L / 4) & (g.x < g.L * 3 / 4)
    average_current = 1 / (g.L / (inside_moat.sum() / g.NG)) * s.v[:, 0].mean()
    expected_current[inside_moat] = average_current
    print(g.dx * expected_current.sum())
    plt.plot(g.x, expected_current, "--", label="expected")
    plt.xticks(g.x)
    plt.legend()
    plt.figure()
    plt.plot(g.x, expected_current - g.current_density[1:-1,0])
    plt.show()


def test_single_particle_longitudinal_deposition(s):
    g = Grid(NG = 7, L=7)
    s.x *= g.dx
    dt = s.c * g.dx
    g.current_density[...] = 0
    longitudinal_current_deposition(g.current_density[:, 0], s.v[:, 0], s.x, dt*np.ones_like(s.x), g.dx, dt, s.q)
    # transversal_current_deposition(g.current_density[:, 1:], s.v, s.x, dt*np.ones_like(s.x), g.dx, dt, s.q)
    collected_longitudinal_weights = g.current_density[:,0].sum()/s.v[0,0]
    # collected_transversal_weights = g.current_density[:,1:].sum()/s.v[0,0]
    def plot_longitudinal():
        fig, ax = plt.subplots()
        ax.scatter(s.x, 0)
        new_positions = s.x + s.v[:,0] * dt
        title = f"x0: {s.x[0]} v: {s.v[0,0]} x1: {new_positions[0]} w: {collected_longitudinal_weights}"
        ax.annotate(title, xy=(new_positions[0], 0), xytext=(s.x[0], 0.3),
                arrowprops=dict(facecolor='black', shrink=0.05, linewidth=0.5), horizontalalignment='right')
        ax.set_xticks(g.x)
        ax.set_xticks(np.arange(0, g.L, g.dx/2), minor=True)
        ax.grid(which='minor', alpha=0.3)
        ax.grid(which='major', alpha=0.7)
        ax.set_title(title)
        ax.scatter(new_positions, 0)
        ax.plot(g.x, g.current_density[1:-1, 0], "go-", alpha=0.7, linewidth=3, label=f"jx")
        ax.legend()
        plt.show()
        return title + " instead of 1"

    def plot_transversal():
        fig, ax = plt.subplots()
        ax.scatter(s.x, 0)
        new_positions = s.x + s.v[:,0] * dt
        title = f'{s.x[0]} to {new_positions[0]}, v={s.v[0,0]}'
        ax.annotate(title, xy=(new_positions[0], 0), xytext=(s.x[0], 0.3),
                    arrowprops=dict(facecolor='black', shrink=0.05, linewidth=0.5), horizontalalignment='right')
        ax.set_xticks(g.x)
        ax.set_xticks(np.arange(0, g.L, g.dx/2), minor=True)
        ax.grid(which='minor', alpha=0.3)
        ax.grid(which='major', alpha=0.7)
        ax.set_title(title)
        ax.scatter(new_positions, 0, "r")
        ax.plot(g.x, g.current_density[1:-1, 1:], "o-", alpha=0.7, linewidth=3, label=f"jx")
        ax.legend()
        plt.show()
        return f"x0: {s.x[0]}\t v: {s.v[0,0]}\t x1: {new_positions[0]}\t w: {collected_longitudinal_weights} instead of 1.0"

    # plot_longitudinal()
    assert np.isclose(collected_longitudinal_weights, 1), plot_longitudinal()
    # print(collected_transversal_weights)
    # assert np.isclose()

    #plt.figure()
    #plt.scatter(s.x, np.zeros_like(s.x))
    #plt.xticks(g.x)
    #plt.grid()
    #for i, label in {1:'y', 2:'z'}.items():
    #    plt.plot(g.x, g.current_density[1:-1, i], alpha=0.7, linewidth=i+3, label=f"j{label}")
    #plt.legend()

if __name__ == '__main__':
    test_single_particle_longitudinal_deposition(Particle(3.01, 0.5, 1, -1))

    for position in (3.01, 3.25, 3.49, 3.51, 3.99):
        for velocity in (0.01, 1, -0.01, -1):
            test_single_particle_longitudinal_deposition(Particle(position, velocity, 1, -1))
