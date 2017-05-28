# coding=utf-8
import pytest
import os

from ..configs import cold_plasma_oscillations
from ..visualization.plotting import plots


@pytest.fixture(scope="module")
def helper_short_simulation():
    if "DISPLAY" not in os.environ.keys():
        print("Not running display test right now.")
        return False
    else:
        run_name = "visualization_test"
        S = cold_plasma_oscillations(run_name, save_data=False)
        return S


def test_static_plots(helper_short_simulation):
    S = helper_short_simulation
    if S:
        try:
            plots(S, save_static=True)
        except:
            assert False, "Failure on saving static plot"

def test_static_plots_frames(helper_short_simulation):

    S = helper_short_simulation
    if S:
        assert True # TODO: finish test


def test_animation(helper_short_simulation):
    S = helper_short_simulation
    if S:
        try:
            plots(S, save_animation=True)
        except:
            assert False, "Failure on saving animation"

