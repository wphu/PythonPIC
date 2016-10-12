from pic3 import *

def l2_norm(reference, test):
    return np.sum((reference-test)**2)/np.sum(reference**2)

def l2_test(reference, test, rtol = 1e-3):
    norm = l2_norm(reference, test)
    print("L2 norm: ", norm)
    return norm < rtol

def test_constant_charge_density():
        NG = 16
        L = 1
        x, dx = np.linspace(0,L,NG, retstep=True,endpoint=False)

        q = 1
        N = 128
        analytical_charge_density = N * q / L

        x_particles = np.linspace(0, L, N, endpoint=False)
        particle_charge = 1

        indices = (x_particles//dx).astype(int)
        print(indices)

        charge_density = charge_density_deposition(x, dx, x_particles, q)
        def plot():
            plt.plot(x, charge_density)
            plt.plot(x, np.ones_like(x)*analytical_charge_density)
            plt.vlines(x, -2*analytical_charge_density, 2*analytical_charge_density)
            plt.show()
            return False
            return "poly test failed for power = {}".format(power)


        assert l2_test(analytical[region_before_last_point], interpolated[region_before_last_point]), plot()
