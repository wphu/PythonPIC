# coding=utf-8

import numpy as np
import pytest

from Species import Species
from helper_functions import get_dominant_mode, show_on_fail
from plotting import plotting
from run_coldplasma import cold_plasma_oscillations


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
        plotting(S, show=show_on_fail, save=False, animate=True))
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
    assert get_dominant_mode(S) == expected_dominant_mode, plotting(S, show=show_on_fail, save=save, animate=animate)


@pytest.mark.parametrize(["proton_mass"], [(100,), (200,), (1836,)])
def test_heavy_protons(proton_mass):
    plasma_frequency = 1
    push_mode = 2
    N_electrons = 1024
    NG = 64
    qmratio = -1
    L = 2 * np.pi
    epsilon_0 = 1
    NT = 150
    proton_charge = 1
    proton_frequency = plasma_frequency / proton_mass ** 0.5
    proton_scaling = abs(proton_mass * proton_frequency ** 2 * L / float(
        proton_mass * N_electrons * epsilon_0))
    # print(proton_frequency, proton_scaling)
    protons = Species(N=N_electrons, q=proton_charge, m=proton_mass, name="protons", NT=NT, scaling=proton_scaling)

    S = cold_plasma_oscillations(f"CO_TWO_SPECIES_{proton_mass}", qmratio=qmratio, plasma_frequency=plasma_frequency,
                                 NG=NG,
                                 N_electrons=N_electrons, push_mode=push_mode, save_data=False, protons=protons)
    # for s in S.list_species:
    #     print(f"{s.name}: DV = {s.velocity_history.max() - s.velocity_history.min()}")
    velocity_ranges = {s.name: s.velocity_history.max() - s.velocity_history.min() for s in S.list_species}
    # print(velocity_ranges)
    velocity_ratio = velocity_ranges['electrons'] / velocity_ranges['protons']
    assert np.isclose(velocity_ratio, proton_mass, rtol=1e-3), (
        f"velocity range ratio is {velocity_ratio}", plotting(S, show=show_on_fail, save=False, animate=True))


@pytest.mark.parametrize(["dt"], [(10,), (100,)])
def test_leapfrog_instability(dt):
    """far above plasma_frequency * dt > 2 the system enters the leapfrog instability
    
    at plasma_frequency * dt ~ 2, we should have odd-even decoupling"""
    plasma_frequency = 1
    push_mode = 2
    N_electrons = 1024
    NG = 64
    qmratio = -1
    NT = 900
    S = cold_plasma_oscillations(f"CO_LEAPFROG", NT=NT, qmratio=qmratio, plasma_frequency=plasma_frequency, NG=NG,
                                 N_electrons=N_electrons, push_mode=push_mode, save_data=False, dt=dt)
    energy_final_to_initial = S.total_energy[-1] / S.total_energy[0]
    assert energy_final_to_initial > 100, (f"Energy gain: {energy_final_to_initial}",
                                           plotting(S, show=show_on_fail, save=False, animate=True))
