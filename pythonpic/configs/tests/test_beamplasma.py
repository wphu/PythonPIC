# coding=utf-8
import pytest
import numpy as np
from ..run_beamplasma import weakbeam_instability, plots
from pythonpic.helper_functions.physics import did_it_thermalize
from . import on_failure
#
@pytest.mark.parametrize(["L", "should_it_thermalize"], [
    (2*np.pi, False),
    (0.2*np.pi, False),
    (100*np.pi, True),
    ])
def test_twostream_likeness(L, should_it_thermalize):
    run_name = f"BP_TWOSTREAM_{N_beam}"
    S = weakbeam_instability(run_name, L=L,
                             save_data=False)
    assert (did_it_thermalize(S)[:2] == should_it_thermalize).all(), ("Incorrect thermalization",
                                                                      plots(S, *on_failure))
