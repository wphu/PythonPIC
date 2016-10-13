from pic3 import *

def l2_norm(reference, test):
    return np.sum((reference-test)**2)/np.sum(reference**2)

def l2_test(reference, test, rtol = 1e-3):
    norm = l2_norm(reference, test)
    print("L2 norm: ", norm)
    return norm < rtol

# def test_constant_charge_density():
#         NG = 16
#         L = 1
#         x, dx = np.linspace(0,L,NG, retstep=True,endpoint=False)

#         q = 1
#         N = 128
#         analytical_charge_density = N * q / L

#         x_particles = np.linspace(0, L, N, endpoint=False)
#         particle_charge = 1

#         charge_density = charge_density_deposition(x, dx, x_particles, q)
#         def plot():
#             plt.plot(x, charge_density)
#             plt.plot(x, np.ones_like(x)*analytical_charge_density)
#             plt.vlines(x, -2*analytical_charge_density, 2*analytical_charge_density)
#             plt.show()
#             return False
#             return "poly test failed for power = {}".format(power)
#         assert l2_test(analytical[region_before_last_point], interpolated[region_before_last_point]), plot()


def test_single_particle():
    NG = 8
    L = 1
    x, dx = np.linspace(0,L,NG, retstep=True,endpoint=False)

    q = 1

    x_particles = np.array([x[3] + dx/2, x[5] + 0.75*dx])
    analytical_charge_density = x_particles.size * q / L

    indices = (x_particles//dx).astype(int)
    print(indices)


    analytical_charge_density = np.array([ 0.,    0.,   0.,    0.5,   0.5,   0.25,  0.75,  0.  ])
    charge_density = charge_density_deposition(x, dx, x_particles, q)
    print("charge density", charge_density)
    def plot():
        plt.plot(x, charge_density, "bo-", label="scattered")
        plt.plot(x, analytical_charge_density, "go-", label="analytical")
        plt.plot(x_particles, q*np.ones_like(x_particles)/x_particles.size, "r*", label="particles")
        plt.legend()
        plt.show()
        return "single particle interpolation is off!"
    assert np.isclose(charge_density, analytical_charge_density).all() , plot()


def test_constant_density():
    NG = 8
    L = 1
    x, dx = np.linspace(0,L,NG, retstep=True,endpoint=False)

    q = 1

    N = 128
    x_particles = np.linspace(0, L, N, endpoint = False)
    analytical_charge_density = x_particles.size * q / L / NG * np.ones_like(x)

    charge_density = charge_density_deposition(x, dx, x_particles, q)
    print("charge density", charge_density)
    def plot():
        plt.plot(x, charge_density, "bo-", label="scattered")
        plt.plot(x, analytical_charge_density, "go-", label="analytical uniform")
        plt.plot(x_particles, q*2/dx*np.ones_like(x_particles), "r*", label="particles")
        plt.legend()
        plt.show()
        return False
    assert np.isclose(charge_density, analytical_charge_density).all() , plot()



def test_boundaries():
    NG = 8
    L = 1
    x, dx = np.linspace(0,L,NG, retstep=True,endpoint=False)

    q = 1

    x_particles = np.array([x[3] + dx/2, x[-1] + 0.25*dx])
    analytical_charge_density = x_particles.size * q / L

    analytical_charge_density = np.array([0.25,  0.,    0.,    0.5,   0.5,   0.,    0.,    0.75])
    charge_density = charge_density_deposition(x, dx, x_particles, q)
    print("charge density", charge_density)
    def plot():
        plt.plot(x, charge_density, "bo-", label="scattered")
        plt.plot(x, analytical_charge_density, "go-", label="analytical")
        plt.plot(x_particles, q*np.ones_like(x_particles)/x_particles.size, "r*", label="particles")
        plt.legend(loc='best')
        plt.show()
        return "single particle interpolation is off!"
    assert np.isclose(charge_density, analytical_charge_density).all() , plot()

if __name__=="__main__":
    test_boundaries()
