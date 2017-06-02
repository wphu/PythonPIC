# coding=utf-8

import pytest

from . import on_failure
from pythonpic.helper_functions.physics import get_dominant_mode
from ..run_coldplasma import cold_plasma_oscillations
from pythonpic.visualization.plotting import plots


@pytest.mark.parametrize("push_mode", range(1, 32, 3))
def test_linear_dominant_mode(push_mode):
    """In the linear mode the """
    plasma_frequency = 1
    N_electrons = 1024
    NG = 64
    qmratio = -1

    run_name = f"CO_LINEAR_{push_mode}"
    S = cold_plasma_oscillations(run_name, qmratio=qmratio, plasma_frequency=plasma_frequency, NG=NG,
                                 N_electrons=N_electrons, push_mode=push_mode, save_data=False).run().postprocess()
    calculated_dominant_mode = get_dominant_mode(S)
    assert calculated_dominant_mode == push_mode, (
        f"got {calculated_dominant_mode} instead of {push_mode}",
        plots(S, *on_failure))


@pytest.mark.parametrize(["N_electrons", "push_amplitude"],
                         [(256, 1e-6), (256, 1e-9)])
def test_kaiser_wilhelm_instability_avoidance(N_electrons, push_amplitude):
    """aliasing effect with particles exactly at or close to grid nodes.
    Particles exactly on grid nodes cause excitation of high modes.
    Even a slight push prevents that."""
    S = cold_plasma_oscillations(f"CO_KWI_STABLE_{N_electrons}_PUSH_{push_amplitude}", save_data=False,
                                 N_electrons=N_electrons, NG=256,
                                 T = 200,
                                 push_amplitude=push_amplitude).run().postprocess()
    assert get_dominant_mode(S) == 1, plots(S, *on_failure)


@pytest.mark.parametrize("N", [128, 256])
def test_kaiser_wilhelm_instability(N):
    __doc__ = test_kaiser_wilhelm_instability_avoidance.__doc__
    S = cold_plasma_oscillations(f"CO_KWI_UNSTABLE_{N}", save_data=False,
                                 N_electrons=N, NG=N,
                                 T = 200,
                                 push_amplitude=0
                                 ).run().postprocess()
    assert get_dominant_mode(S) > 5, plots(S, *on_failure)





