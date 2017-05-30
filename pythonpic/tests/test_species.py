# coding=utf-8
import pytest
import numpy as np
from ..classes.species import n_saved_particles

@pytest.fixture(params=np.logspace(0, 6, 12, dtype=int))
def n_available(request):
    return request.param

max_saved = n_available

def test_n_saved(n_available, max_saved):
    save_every_n, n_saved = n_saved_particles(n_available, max_saved)

    assert n_saved <= max_saved
    assert np.arange(n_available)[::save_every_n].size == n_saved

def test_n_saved_equal(n_available):
    save_every_n, n_saved = n_saved_particles(n_available, n_available)

    assert n_saved == n_available
