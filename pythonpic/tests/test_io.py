# coding=utf-8
import pytest
import numpy as np
from ..classes import Grid, Particle, Simulation, load_simulation
from ..helper_functions.file_io import config_filename

@pytest.fixture(scope="module")
def test_io_helper():
    grid = Grid(1, 1, 100)
    species = Particle(grid, 0.5, 0.1)
    filename = "io_test"
    sim = Simulation(grid, [species], filename = filename)
    sim.grid_species_initialization()
    sim.run().save_data()

    sim2 = load_simulation(config_filename(filename))
    return sim, sim2

def test_grid_io(test_io_helper):
    sim, sim2 = test_io_helper
    g = sim.grid
    g.postprocess()
    g2 = sim2.grid

    assert np.allclose(g.x, g2.x)
    assert np.allclose(g.c, g2.c)
    assert np.allclose(g.epsilon_0, g2.epsilon_0)
    assert np.allclose(g.dt, g2.dt)
    assert np.allclose(g.T, g2.T)
    assert np.allclose(g.NT, g2.NT)
    assert np.allclose(g.k, g2.k)
    assert np.allclose(g.charge_density_history, g2.charge_density_history)
    assert np.allclose(g.current_density_history, g2.current_density_history)
    assert np.allclose(g.electric_field_history, g2.electric_field_history)
    assert np.allclose(g.magnetic_field_history, g2.magnetic_field_history)
    assert np.allclose(g.postprocessed, g2.postprocessed)

def test_species_io(test_io_helper):
    sim, sim2 = test_io_helper
    s = sim.list_species[0]
    s.postprocess()
    s2 = sim2.list_species[0]

    assert np.allclose(s.q, s2.q)
    assert np.allclose(s.m, s2.m)
    assert np.allclose(s.scaling, s2.scaling)
    assert np.allclose(s.eff_q, s2.eff_q)
    assert np.allclose(s.eff_m, s2.eff_m)
    assert np.allclose(s.dt, s2.dt)
    assert np.allclose(s.NT, s2.NT)
    assert np.allclose(s.c, s2.c)
    assert np.allclose(s.position_history, s2.position_history)
    assert np.allclose(s.velocity_history, s2.velocity_history)
    assert np.allclose(s.velocity_mean_history, s2.velocity_mean_history)
    assert np.allclose(s.velocity_std_history, s2.velocity_std_history)
    assert np.allclose(s.postprocessed, s2.postprocessed)
    assert np.allclose(s.kinetic_energy_history, s2.kinetic_energy_history)
    with pytest.raises(AssertionError):
        assert np.allclose(s.v, s2.v)
    with pytest.raises(AssertionError):
        assert np.allclose(s.x, s2.x)
