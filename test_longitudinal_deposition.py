import numpy as np
from matplotlib import pyplot as plt

from Grid import Grid
from Species import Species
from algorithms_interpolate import longitudinal_current_deposition


def test_longitudinal_deposition():
    for power in range(2, 8):
        N = 10 ** power
        s = Species(1 / N, 1, N, "test particles", 1, 1, 1)
        g = Grid()
        s.distribute_uniformly(g.L, 1e-6, g.L / 4, g.L / 4)
        s.v[:, 0] = np.random.normal(scale=0.1, size=N)
        # s.v[:,0] = s.c*0.9
        dt = g.dx

        longitudinal_current_deposition(g.current_density[:, 0], s.v[:, 0], s.x, dt, g.dx, dt, s.q)
        plt.plot(g.x, g.current_density[1:-1, 0], label=N)
    expected_current = np.zeros_like(g.x)
    inside_moat = (g.x > g.L / 4) & (g.x < g.L * 3 / 4)
    average_current = 1 / (g.L / (inside_moat.sum() / g.NG)) * s.v[:, 0].mean()
    expected_current[inside_moat] = average_current
    plt.plot(g.x, expected_current, label="expected")
    plt.xticks(g.x)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_longitudinal_deposition()
