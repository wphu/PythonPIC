from Species import Species
from Grid import Grid
import algorithms_density_profiles
import pytest
import numpy as np
import matplotlib.pyplot as plt

@pytest.fixture(params=np.linspace(0.01, 0.5, 4))
def _fraction(request):
    return request.param

_second_fraction = _fraction

@pytest.fixture(params=algorithms_density_profiles.profiles.keys())
def _profile(request):
    return request.param

def test_density(_fraction, _second_fraction, _profile):
    N = 1000
    g = Grid(10, 10)
    s = Species(1, 1, N)

    moat_length = g.L * _fraction
    ramp_length = g.L * _second_fraction
    plasma_length = g.L - moat_length - ramp_length
    plasma_length = plasma_length if plasma_length > ramp_length else ramp_length
    s.distribute_nonuniformly(g.L, moat_length, ramp_length, plasma_length, profile=_profile)
    minimal_warn = f"min particle x: {s.x.min()} < moat_length {moat_length}"
    maximal_warn = f"max particle x: {s.x.max()} > plasma region {moat_length + plasma_length}"

    def plot(warn):
        # plt.hist(s.x, g.x)
        # plt.show()
        print(_fraction, _second_fraction, plasma_length/g.L)
        return warn
    assert (s.x > moat_length).all(), plot(minimal_warn)
    assert (s.x <= moat_length + plasma_length).all(), plot(maximal_warn)
