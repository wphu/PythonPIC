import matplotlib.pyplot as plt
from scatter import charge_density_deposition
import numpy as np

def test_sine_perturbation_effect(amplitude=0.001):
    g = Grid(L=1, NG=32)
    particles = Species(1, 1, 128)
    particles.distribute_uniformly(g.L)
    particles.sinusoidal_position_perturbation(amplitude, 1, g.L)

    g.gather_charge([particles])
    def plots():
        plt.hist(particles.x, bins=g.x)
        plt.plot(g.x, g.charge_density, "bo-", label="scattered")
        plt.vlines(g.x, 3, 5)
        plt.plot(particles.x, np.ones(128)*4, "ro")
        plt.show()
    assert True, plots()

def test_single_particle(plotting=False):
    NG = 8
    L = 1
    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)

    q = 1

    x_particles = np.array([x[3] + dx / 2, x[5] + 0.75 * dx])
    analytical_charge_density = x_particles.size * q / L

    indices = (x_particles // dx).astype(int)
    print(indices)

    analytical_charge_density = np.array([0., 0., 0., 0.5, 0.5, 0.25, 0.75, 0.])
    charge_density = charge_density_deposition(x, dx, x_particles, q)
    print("charge density", charge_density)

    def plot():
        plt.plot(x, charge_density, "bo-", label="scattered")
        plt.plot(x, analytical_charge_density, "go-", label="analytical")
        plt.plot(x_particles, q * np.ones_like(x_particles) / x_particles.size, "r*", label="particles")
        plt.legend()
        plt.show()
        return "single particle interpolation is off!"
    if plotting:
        plot()
    assert np.isclose(charge_density, analytical_charge_density).all(), plot()


def test_constant_density(plotting=False):
    NG = 8
    L = 1
    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)

    q = 1

    N = 128
    x_particles = np.linspace(0, L, N, endpoint=False)
    analytical_charge_density = x_particles.size * q / L / NG * np.ones_like(x)

    charge_density = charge_density_deposition(x, dx, x_particles, q)
    print("charge density", charge_density)

    def plot():
        plt.plot(x, charge_density, "bo-", label="scattered")
        plt.plot(x, analytical_charge_density, "go-", label="analytical uniform")
        plt.plot(x_particles, q * 2 / dx * np.ones_like(x_particles), "r*", label="particles")
        plt.legend()
        plt.show()
        return False
    if plotting:
        plot()
    assert np.isclose(charge_density, analytical_charge_density).all(), plot()


def test_boundaries(plotting=False):
    NG = 8
    L = 1
    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)

    q = 1

    x_particles = np.array([x[3] + dx / 2, x[-1] + 0.25 * dx])
    analytical_charge_density = x_particles.size * q / L

    analytical_charge_density = np.array([0.25, 0., 0., 0.5, 0.5, 0., 0., 0.75])
    charge_density = charge_density_deposition(x, dx, x_particles, q)
    print("charge density", charge_density)

    def plot():
        plt.plot(x, charge_density, "bo-", label="scattered")
        plt.plot(x, analytical_charge_density, "go-", label="analytical")
        plt.plot(x_particles, q * np.ones_like(x_particles) / x_particles.size, "r*", label="particles")
        plt.legend(loc='best')
        plt.show()
        return "single particle interpolation is off!"
    if plotting:
        plot()
    assert np.isclose(charge_density, analytical_charge_density).all(), plot()

if __name__ == "__main__":
    test_sine_perturbation_effect()
