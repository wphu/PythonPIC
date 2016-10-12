from pic3 import *

def test_poly():
        NG = 16
        L = 1

        x, dx = np.linspace(0,L,NG, retstep=True,endpoint=False)
        # charge_density = np.zeros_like(x)

        N = 128
        x_particles = np.linspace(0, L, N, endpoint=False)
        particle_charge = 1

        indices = (x_particles/dx).astype(int)
        print(indices)

        for electric_field in np.ones_like(x), x, x**2:
            interpolated = interpolateField(x_particles, electric_field, x, dx)
            
            region_before_last_point = x_particles < x.max()
            grid_coeffs = np.polyfit(x, electric_field, 3)
            part_coeffs = np.polyfit(x_particles[region_before_last_point],
                                     interpolated[region_before_last_point], 3)
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
