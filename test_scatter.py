from pic3 import *
from helper_functions import l2_norm, l2_test
import matplotlib.pyplot as plt

def new_test_single_particle():
    g = Grid(L=1, NG=8)
    particles = Species(1, 1, 2)
    particles.x = np.array([g.x[1] + g.dx / 2, g.x[5] + 0.75 * g.dx])

    g.gather_charge(particles)
    plt.plot(g.x, g.charge_density, "bo-", label="scattered")
    plt.show()

def test_single_particle():
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
    assert np.isclose(charge_density, analytical_charge_density).all(), plot()


def test_constant_density():
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
    assert np.isclose(charge_density, analytical_charge_density).all(), plot()


def test_boundaries():
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
    assert np.isclose(charge_density, analytical_charge_density).all(), plot()

if __name__ == "__main__":
    new_test_single_particle()
