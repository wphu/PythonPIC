# coding=utf-8
import numpy as np
import pytest

from ..algorithms import density_profiles
from ..classes import Species, Grid


@pytest.fixture(params=np.linspace(0.01, 0.5, 3), scope='module')
def _fraction(request):
    return request.param


_second_fraction = _fraction

@pytest.fixture(params=np.linspace(1000, 15000, 3, dtype=int), scope='module')
def _N(request):
    return request.param

@pytest.fixture(params=density_profiles.profiles.keys(), scope='module')
def _profile(request):
    return request.param

# noinspection PyUnresolvedReferences
@pytest.fixture(scope='module')
def test_density_helper(_fraction, _second_fraction, _profile, _N):

    g = Grid(1, 100, 100)
    s = Species(1, 1, _N, g)

    moat_length = g.L * _fraction
    ramp_length = g.L * _second_fraction
    plasma_length = 2*ramp_length
    # plasma_length = plasma_length if plasma_length > ramp_length else ramp_length
    s.distribute_nonuniformly(g.L, moat_length, ramp_length, plasma_length, profile=_profile)
    return s, g, moat_length, plasma_length, _profile

def test_particles_out_bounds(test_density_helper):
    s, g, moat_length, plasma_length, profile = test_density_helper
    minimal_warn = f"min particle x: {s.x.min()} < moat_length {moat_length}"
    maximal_warn = f"max particle x: {s.x.max()} > plasma region {moat_length + plasma_length}"

    def plot(warn):
        print(_fraction, _second_fraction, plasma_length / g.L)
        return warn

    assert (s.x > moat_length).all(), plot(minimal_warn)
    assert (s.x <= moat_length + plasma_length).all(), plot(maximal_warn)

def test_particle_conservation(test_density_helper):
    s, g, moat_length, plasma_length, profile = test_density_helper

    def plot():
        # # plt.plot(dense_x, linear_distribution_function, label="Distribution function")
        # plt.scatter(s.x, s.x, c="g", alpha=0.1, label="Particles")
        # plt.hist(s.x, g.x, label="Particle density", alpha=0.5);
        # # plt.xlim(0, g.L)
        # # plt.xticks(g.x)
        # # plt.gca().xaxis.set_ticklabels([])
        title = f"{difference} particles disappeared at N: {s.N} with {profile} profile!"
        # plt.title(title)
        # plt.legend()
        # plt.show()
        return  title
    difference = s.N - s.x.size
    assert difference == 0, plot()
