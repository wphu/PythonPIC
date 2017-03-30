# coding=utf-8

import pytest

from plotting import plotting
from run_beamplasma import weakbeam_instability


@pytest.mark.parametrize(["N_beam", ], [
    (128,),
    (512,),
    (2048,),
    ])
def test_twostream_likeness(N_beam):
    run_name = f"BP_TWOSTREAM_{N_beam}"
    S = weakbeam_instability(run_name, N_beam=N_beam,
                             save_data=False)
    plotting(S, show=True, save=False, animate=True)
    assert False
