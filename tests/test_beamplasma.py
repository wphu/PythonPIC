# coding=utf-8

import pytest

from helper_functions import did_it_thermalize
from plotting import plotting
from run_beamplasma import weakbeam_instability


@pytest.mark.parametrize(["N_beam", "should_it_thermalize"], [
    (128, False),
    (512, False),
    (2048, True),
    ])
def test_twostream_likeness(N_beam, should_it_thermalize):
    run_name = f"BP_TWOSTREAM_{N_beam}"
    S = weakbeam_instability(run_name, N_beam=N_beam,
                             save_data=False)
    assert (did_it_thermalize(S) == should_it_thermalize).all(), plotting(S, show=True, save=False, animate=True)
