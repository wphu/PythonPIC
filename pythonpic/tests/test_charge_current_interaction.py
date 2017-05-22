import pytest
import matplotlib.pyplot as plt

from ..classes import Grid, Species, Simulation


@pytest.fixture(params=[1], scope='module')
def L(request):
    return request.param
@pytest.fixture(params=[10], scope='module')
def NG(request):
    return request.param
@pytest.fixture(params=[100], scope='module')
def NP(request):
    return request.param

def test(L, NG, NP):
    NT = 2
    c = 1

    g = Grid(L, NG, NT=NT)
    s = Species(1, 1, NP, NT=NT)
    s.v[:,0] = 0.99
    s.distribute_uniformly(g.L, 0, g.L/4, g.L/4)
    dt = g.dx * c
    sim = Simulation(NT, dt, [s], g)
    sim.grid_species_initialization()
    print(g.charge_density)
    first_charge = g.charge_density.copy()
    saved_current = g.current_density.copy()
    s.push(lambda x: 0, dt)
    g.gather_charge([s], 0)
    second_charge = g.charge_density.copy()

    charge_change = second_charge - first_charge
    print(charge_change)
    print(g.current_density[:,0])
    def plot():
        plt.plot(g.x, charge_change[1:-1]/dt, label="Change in charge")
        plt.plot(g.x, -saved_current[1:-1, 0], label="Current")
        plt.legend()
        plt.show()
    plot()
if __name__ == '__main__':
    test(1, 10, 100)