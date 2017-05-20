# coding=utf-8
import numpy as np
import pytest
from pythonpic.classes.species import Species

from pythonpic.algorithms import density_profiles
from pythonpic.classes.grid import Grid


@pytest.fixture(params=np.linspace(0.01, 0.5, 4))
def _fraction(request):
    return request.param


_second_fraction = _fraction

@pytest.fixture(params=np.linspace(10, 15000, 5, dtype=int))
def _N(request):
    return request.param

@pytest.fixture(params=density_profiles.profiles.keys())
def _profile(request):
    return request.param

# noinspection PyUnresolvedReferences
@pytest.fixture()
def test_density_helper(_fraction, _second_fraction, _profile, _N):
    g = Grid(10, 10)
    s = Species(1, 1, _N, "particles")

    moat_length = g.L * _fraction
    ramp_length = g.L * _second_fraction
    plasma_length = g.L - moat_length - ramp_length
    plasma_length = plasma_length if plasma_length > ramp_length else ramp_length
    s.distribute_nonuniformly(g.L, moat_length, ramp_length, plasma_length, profile=_profile)
    return s, moat_length, plasma_length

# def test_particles_out_bounds(test_density_helper):
#     s, moat_length, plasma_length = test_density_helper
#     minimal_warn = f"min particle x: {s.x.min()} < moat_length {moat_length}"
#     maximal_warn = f"max particle x: {s.x.max()} > plasma region {moat_length + plasma_length}"
#
#     def plot(warn):
#         print(_fraction, _second_fraction, plasma_length / g.L)
#         return warn
#
#     assert (s.x > moat_length).all(), plot(minimal_warn)
#     assert (s.x <= moat_length + plasma_length).all(), plot(maximal_warn)

def test_particle_conservation(test_density_helper):
    s, _, _ = test_density_helper
    difference = s.N - s.x.size
    assert difference == 0, f"{difference} particles disappeared at N: {s.N}!"
