# coding=utf-8
import numpy as np
import pytest
import matplotlib.pyplot as plt

from ..algorithms import density_profiles
from ..classes import Species, Grid, Simulation
from ..visualization.time_snapshots import SpatialPerturbationDistributionPlot

@pytest.fixture(params=np.linspace(0.1, 0.5, 3), scope='module')
def _fraction(request):
    return request.param


_second_fraction = _fraction

@pytest.fixture(params=np.linspace(300, 4000, 3, dtype=int), scope='module')
def _N(request):
    return request.param

@pytest.fixture(params=density_profiles.profiles.keys(), scope='module')
def _profile(request):
    return request.param

# noinspection PyUnresolvedReferences
@pytest.fixture()
def test_density_helper(_fraction, _second_fraction, _profile, _N):

    g = Grid(1, 100, 100)
    s = Species(1, 1, _N, g)

    moat_length = g.L * _fraction
    ramp_length = g.L * _second_fraction
    plasma_length = 2*ramp_length
    # plasma_length = plasma_length if plasma_length > ramp_length else ramp_length
    s.distribute_nonuniformly(g.L, moat_length, ramp_length, plasma_length, profile=_profile)
    return s, g, moat_length, plasma_length, _profile

@pytest.mark.parametrize("std", [0.0001])
def test_fitness(test_density_helper, std):
    s, g, moat_length, plasma_length, profile = test_density_helper
    assert (s.x > moat_length).all(), "Particles running out the right side!"
    assert (s.x <= moat_length + plasma_length).all(), "Particles running out the left side!"
    # particle conservation
    assert s.N == s.x.size, "Particles are not conserved!"
    sim = Simulation(g, [s])
    s.gather_density()
    s.save_particle_values(0)
    s.random_position_perturbation(std)
    s.gather_density()
    s.save_particle_values(1)
    def plots():
        fig, ax = plt.subplots()
        fig.suptitle(std)
        plot = SpatialPerturbationDistributionPlot(sim, ax)
        plot.update(1)
        # plot2 = SpatialDistributionPlot(sim, ax)
        # plot2.update(0)
        plt.show()
    assert np.allclose(s.density_history[0], s.density_history[1], atol=1e-3), plots()


