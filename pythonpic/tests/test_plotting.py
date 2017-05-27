import pytest

from ..configs import cold_plasma_oscillations
from ..visualization.plotting import plots


@pytest.fixture(scope="module")
def helper_short_simulation():
    run_name = "visualization_test"
    S = cold_plasma_oscillations(run_name, save_data=False)
    return S


def test_static_plots(helper_short_simulation):
    S = helper_short_simulation
    try:
        plots(S, False, True, False, False)
    except:
        assert False, "Failure on saving static plot"


def test_animation(helper_short_simulation):
    S = helper_short_simulation
    try:
        plots(S, False, False, False, True)
    except:
        assert False, "Failure on saving animation"

