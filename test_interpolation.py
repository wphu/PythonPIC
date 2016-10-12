from pic3 import *

def test_poly():
        NG = 128
        L = 1

        x, dx = np.linspace(0,L,NG, retstep=True,endpoint=False)
        # charge_density = np.zeros_like(x)

        x_particles = np.linspace(0, L, endpoint=False)
        particle_charge = 1

        for electric_field in np.ones_like(x), x, x**2:
            interpolated = interpolateField(x_particles, electric_field, x, dx)
            def plot():
                plt.plot(x, electric_field)
                plt.plot(x_particles, interpolated, "go-")
                plt.show()

            grid_coeffs = np.polyfit(x, electric_field, 3)
            part_coeffs = np.polyfit(x_particles, interpolated, 3)
            # print(grid_coeffs, part_coeffs)
            assert np.isclose(grid_coeffs, part_coeffs).all(), plot()
        # charge_density = charge_density_deposition(x, dx, x_particles, particle_charge)

if __name__=="__main__":
    test_poly()
