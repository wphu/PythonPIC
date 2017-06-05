# coding=utf-8
import pytest
import numpy as np
import matplotlib.pyplot as plt
from pythonpic.visualization.time_snapshots import SpatialPerturbationDistributionPlot

from . import on_failure
from ..run_laser import laser

@pytest.mark.parametrize("std", [1e-3, 1e-4, 1e-6])
def test_stability(std):
    S = laser("stability_test", 10000, 0, 0, std).test_run()
    def plots():
        fig, ax = plt.subplots()
        fig.suptitle(std)
        plot = SpatialPerturbationDistributionPlot(S, ax)
        plot.update(S.NT-1)
        plt.show()
    s = S.list_species[0]
    plots()
    assert np.allclose(s.density_history[0], s.density_history[-1]), plots()
