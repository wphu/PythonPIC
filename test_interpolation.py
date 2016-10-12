from pic3 import *

def test_poly():
        NG = 16
        L = 1

        x, dx = np.linspace(0,L,NG, retstep=True,endpoint=False)
        # charge_density = np.zeros_like(x)

        N = 128
        x_particles = np.linspace(0, L, N, endpoint=False)
        particle_charge = 1

        for power in range(3):
            electric_field_function = lambda x: x**power
            electric_field = electric_field_function(x)

            interpolated = interpolateField(x_particles, electric_field, x, dx)
            analytical = electric_field_function(x_particles)

            region_before_last_point = x_particles < x.max()
            grid_coeffs = np.polyfit(x, electric_field, 3)
            part_coeffs = np.polyfit(x_particles[region_before_last_point],
                                     interpolated[region_before_last_point], 3)
            def plot():
                plt.plot(x, electric_field, lw=5)
                plt.plot(x_particles, interpolated, "go-")
                plt.vlines(x, electric_field.min(), electric_field.max())
                plt.show()
                return """grid: {:.3e}x^3 + {:.3e}x^2 + {:.3e}x^1 + {:.3e}
                          particles: {:.3e}x^3 + {:.3e}x^2 + {:.3e}x^1 + {:.3e}""".format(*grid_coeffs, *part_coeffs)


            assert np.isclose(analytical[region_before_last_point], interpolated[region_before_last_point]).all(), plot()
        # charge_density = charge_density_deposition(x, dx, x_particles, particle_charge)


def test_periodic():
        NG = 16
        L = 1

        x, dx = np.linspace(0,L,NG, retstep=True,endpoint=False)
        # charge_density = np.zeros_like(x)

        N = 128
        x_particles = np.linspace(0, L, N, endpoint=False)
        particle_charge = 1

        indices = (x_particles/dx).astype(int)
        print(indices)

        for electric_field in np.sin(2*np.pi*x), np.cos(2*np.pi*x):
            interpolated = interpolateField(x_particles, electric_field, x, dx)
            
            grid_coeffs = np.polyfit(x, electric_field, 3)
            part_coeffs = np.polyfit(x_particles, interpolated, 3)
            def plot():
                plt.plot(x, electric_field, lw=5)
                plt.plot(x_particles, interpolated, "go-")
                plt.vlines(x, electric_field.min(), electric_field.max())
                plt.show()
                return grid_coeffs, part_coeffs

            assert np.isclose(grid_coeffs, part_coeffs).all(), plot()
        # charge_density = charge_density_deposition(x, dx, x_particles, particle_charge)

if __name__=="__main__":
    test_poly()
