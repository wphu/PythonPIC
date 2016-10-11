from pic3 import *

# def charge_density_deposition(x, dx, x_particles, particle_charge):
#     """Calculates the charge density on a 1D grid given an array of charged particle positions.
#     x: array of grid positions
#     dx: grid positions step
#     x_particles: array of particle positions on the grid.
#         make sure this is 0 < x_particles < L
#     """
#     assert ((x_particles<L).all() and (0<x_particles).all()), (x_particles, x_particles[x_particles>L])
#     indices_on_grid = (x_particles/dx).astype(int)
#
#     charge_density=np.zeros_like(x)
#     for (i, index), xp in zip(enumerate(indices_on_grid), x_particles):
#         charge_density[index]+=particle_charge * (dx+x[index]-xp)/dx
#         charge_density[(index+1)%(NG)] += particle_charge * (xp - x[index])/dx
#     return charge_density



def test_charge_density():
    NG = 128
    L = 1

    x, dx = np.linspace(0,L,NG, retstep=True,endpoint=False)
    # charge_density = np.zeros_like(x)

    x_particles = np.linspace(0, L, endpoint=False)
    particle_charge = 1


    charge_density = charge_density_deposition(x, dx, x_particles, particle_charge)

    def plots():
        plt.plot(x, charge_density)
        plt.scatter(x_particles, np.ones_like(x_particles))
        plt.show()

    assert True, plots()

if __name__=="__main__":
    test_charge_density()
