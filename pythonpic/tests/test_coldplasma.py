# coding=utf-8

import numpy as np
import pytest

from ..algorithms.helper_functions import get_dominant_mode, show_on_fail
from ..classes.species import Species
from ..configs.run_coldplasma import cold_plasma_oscillations
from ..visualization.plotting import plots


@pytest.mark.parametrize("push_mode", range(1, 32, 3))
def test_linear_dominant_mode(push_mode):
    plasma_frequency = 1
    N_electrons = 1024
    NG = 64
    qmratio = -1

    run_name = f"CO_LINEAR_{push_mode}"
    S = cold_plasma_oscillations(run_name, qmratio=qmratio, plasma_frequency=plasma_frequency, NG=NG,
                                 N_electrons=N_electrons, push_mode=push_mode, save_data=False)
    calculated_dominant_mode = get_dominant_mode(S)
    assert calculated_dominant_mode == push_mode, (
        f"got {calculated_dominant_mode} instead of {push_mode}",
        plots(S, show=show_on_fail, save=False, animate=True))
    return S


@pytest.mark.parametrize(["N_electrons", "expected_dominant_mode", "push_amplitude"],
                         [(32, 7, 0), (33, 1, 0), (32, 1, 1e-6)])
def test_kaiser_wilhelm(N_electrons, expected_dominant_mode, push_amplitude):
    """aliasing effect with particles exactly at or close to grid nodes.
    Particles exactly on grid nodes cause excitation of high modes.
    Even a slight push prevents that."""
    S = cold_plasma_oscillations(f"CO_KW_{N_electrons}{'_PUSH' if push_amplitude != 0 else ''}", save_data=False,
                                 N_electrons=N_electrons, NG=16,
                                 push_amplitude=push_amplitude)
    show, save, animate = False, False, True
    assert get_dominant_mode(S) == expected_dominant_mode, plots(S, show=show_on_fail, save=save, animate=animate)



