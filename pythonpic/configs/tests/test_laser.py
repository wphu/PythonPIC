# coding=utf-8
import pytest
import numpy as np
import matplotlib.pyplot as plt
from pythonpic.helper_functions.helpers import make_sure_path_exists
from pythonpic.visualization.time_snapshots import SpatialPerturbationDistributionPlot

from . import on_failure
from ..run_laser import laser

@pytest.mark.parametrize("std", [0])
def test_stability(std):
    S = laser("stability_test", 1000, 0, 0, std).test_run()
    def plots():
        fig, ax = plt.subplots()
        fig.suptitle(std)
        plot = SpatialPerturbationDistributionPlot(S, ax)
        plot.update(S.NT-1)
        make_sure_path_exists(S.filename)
        fig.savefig(S.filename.replace(".hdf5", ".png"))
        plt.close(fig)
    s = S.list_species[0]
    assert np.allclose(s.density_history[0], s.density_history[-1]), plots()
