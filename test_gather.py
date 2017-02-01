import numpy as np
from helper_functions import l2_test
from grid_algorithms import interpolateField
import matplotlib.pyplot as plt

def test_poly(plotting=False):
    NG = 16
    NG_plot = 500
    L = 1

    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
    # charge_density = np.zeros_like(x)

    N = 128
    x_particles = np.linspace(0, L, N, endpoint=False)

    for power in range(6):
        electric_field_function = lambda x: x**power
        electric_field = electric_field_function(x)

        interpolated = interpolateField(x_particles, electric_field, x, dx)
        analytical = electric_field_function(x_particles)

        region_before_last_point = x_particles < x.max()

        def plot():
            x_plot = np.linspace(0, L, NG_plot, endpoint=False)
            electric_field_plot = electric_field_function(x_plot)
            plt.plot(x_plot, electric_field_plot, lw=5)
            plt.plot(x, electric_field)
            plt.plot(x_particles, interpolated, "go-")
            plt.vlines(x, electric_field.min(), electric_field.max())
            plt.show()
            return "poly test failed for power = {}".format(power)
        if plotting:
            plot()

        assert l2_test(analytical[region_before_last_point], interpolated[region_before_last_point]), plot()
    # charge_density = charge_density_deposition(x, dx, x_particles, particle_charge)


def test_periodic(plotting=False):
    NG = 16
    NG_plot = 500
    L = 1

    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
    # charge_density = np.zeros_like(x)

    N = 128
    x_particles = np.linspace(0, L, N, endpoint=False)
    particle_charge = 1

    for func in lambda x: np.sin(2 * np.pi * x), lambda x: np.cos(2 * np.pi * x):
        electric_field = func(x)
        interpolated = interpolateField(x_particles, electric_field, x, dx)
        analytical = func(x_particles)

        def plot():
            x_plot = np.linspace(0, L, NG_plot, endpoint=False)
            electric_field_plot = func(x_plot)
            plt.plot(x_plot, electric_field_plot, lw=5)
            plt.plot(x, electric_field)
            plt.plot(x_particles, interpolated, "go-")
            plt.vlines(x, electric_field.min(), electric_field.max())
            plt.show()
            return "periodic test failure"
        if plotting:
            plot()

        assert l2_test(interpolated, analytical), plot()
    # charge_density = charge_density_deposition(x, dx, x_particles, particle_charge)


def test_single_particle(plotting=False):
    """tests interpolation of field to particles:
        at cell boundary
        at hall cell
        at 3/4 cell
        at end of simulation region (PBC)
    """
    NG = 16
    L = 1

    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
    x_particles = np.array([x[3], x[6] + dx / 2, x[9] + 0.75 * dx, x[-1] + dx / 2])

    for power in range(2,3):
        electric_field_function = lambda x: x**power
        electric_field = electric_field_function(x)

        interpolated = interpolateField(x_particles, electric_field, x, dx)
        analytical = electric_field_function(x_particles)
        analytical[-1] = (electric_field[0] + electric_field[-1]) / 2

        def plot():
            plt.plot(x, electric_field)
            plt.plot(x_particles, interpolated, "go-")
            plt.vlines(x, electric_field.min(), electric_field.max())
            plt.show()
            return "poly test failed for power = {}".format(power)
        if plotting:
            plot()

        assert l2_test(analytical, interpolated), plot()

if __name__ == "__main__":
    test_single_particle()
