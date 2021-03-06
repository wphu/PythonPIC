# coding=utf-8
import pytest
import numpy as np
from ..run_beamplasma import weakbeam_instability, plots
from pythonpic.helper_functions.physics import did_it_thermalize
from . import on_failure
#
@pytest.mark.parametrize(["L", "T", "should_it_thermalize"], [
    (2*np.pi, 60, False),
    (0.2*np.pi, 60, False),
    ])
def test_twostream_likeness(L, T, should_it_thermalize):
    run_name = f"BP_TWOSTREAM_{L}"
    S = weakbeam_instability(run_name, L=L, T=T).test_run()
    assert (did_it_thermalize(S)[:2] == should_it_thermalize).all(), ("Incorrect thermalization",
                                                                      plots(S, *on_failure, alpha=0.5))
