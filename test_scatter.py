import matplotlib.pyplot as plt
from scatter import charge_density_deposition
import numpy as np
from Grid import Grid
from Species import Species

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

def test_uniform_current_deposition(plotting=False):
    """
     N particles
     charge q
     velocity v
    """


    g = Grid()
    p = Species(1, 1, 128, "p")
    p.v[:, 0] = 0
    p.v[:, 1] = -1
    p.v[:, 2] = 1
    p.distribute_uniformly(g.L, g.dx/1000*np.pi)
    g.current_density = g.gather_current([p])
    if plotting:
        plt.plot(g.x, g.current_density)
        plt.show()

    predicted_values = p.q*p.v.sum(axis=0)/g.L*g.dx
    for dim, val in zip(range(3), predicted_values):
        assert np.isclose(g.current_density[:,dim], val).all(), (g.current_density[0,:], predicted_values)

def test_nonuniform_current_deposition(plotting=False):
    """
     N particles
     charge q
     velocity v
    """

    g = Grid()
    p = Species(1, 1, 128, "p")
    dims = np.arange(3)
    p.v[:, dims] = np.arange(p.N)[:,np.newaxis]**dims[np.newaxis,:]
    p.distribute_uniformly(g.L, g.dx/1000*np.pi)
    g.current_density = g.gather_current([p])
    if plotting:
        plt.plot(g.x, g.current_density)
        plt.show()

    predicted_values = p.q*p.v.sum(axis=0)/g.L*g.dx
    for dim, val in zip(dims, predicted_values):
        indices = np.arange(1, g.NG)
        fit = np.polyfit(g.x[indices], g.current_density[indices,dim], 2)
        fit /= np.linalg.norm(fit)
        print(fit)
        if plotting:
            plt.plot(g.x, np.polyval(fit, g.x))
            plt.plot(g.x, g.current_density[:,dim])
            plt.show()
        assert np.isclose(fit[2-dim], 1, rtol=1e-4), (fit[dim])


if __name__ == "__main__":
    test_nonuniform_current_deposition(True)
