# coding=utf-8
import numpy as np
import pytest
import matplotlib.pyplot as plt

from Grid import Grid
from Species import Species
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


def test_single_particle_deposition():
    s = Species(1, 1, 4, "test particle", 1, 1, 1)
    g = Grid(NG = 16, L=16)

    s.x[:] = np.array([1 + 0.49, 5 + 0.49, 8 + 0.51, 13 + 0.51]) * g.dx
    s.v[:, 0] = np.array([1, -1, 1, -1], dtype=float)

    s.v[:, 1] = +1
    s.v[:, 2] = -1
    dt = s.c * g.dx
    fig, ax = plt.subplots()
    pax = ax.twinx()
    pax.scatter(s.x, s.x)
    new_positions = s.x + s.v[:,0] * dt
    pax.scatter(s.x, new_positions)
    for xn, xnp, v in zip(s.x, new_positions, s.v[:,0]):
        print(xn, xnp, v)
        pax.annotate(f"{xn} to {xnp}, v {v}", (xn, xnp), (0, 0), arrowprops={'arrowstyle':'->'}) 
    ax.set_xticks(np.arange(0, g.L, g.dx/2))
    ax.grid()
    longitudinal_current_deposition(g.current_density[:, 0], s.v[:, 0], s.x, dt, g.dx, dt, s.q)
    #transversal_current_deposition(g.current_density[:, 1:], s.v, s.x, dt*np.ones_like(s.x), g.dx, dt, s.q)
    ax.plot(g.x, g.current_density[1:-1, 0], "go-", alpha=0.7, linewidth=3, label=f"jx")
    ax.legend()

    #plt.figure()
    #plt.scatter(s.x, np.zeros_like(s.x))
    #plt.xticks(g.x)
    #plt.grid()
    #for i, label in {1:'y', 2:'z'}.items():
    #    plt.plot(g.x, g.current_density[1:-1, i], alpha=0.7, linewidth=i+3, label=f"j{label}")
    #plt.legend()
    plt.show()

if __name__ == '__main__':
    test_single_particle_deposition()
    # test_longitudinal_deposition(7)
