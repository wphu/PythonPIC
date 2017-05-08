""" Run wave propagation"""
# coding=utf-8
import numpy as np
import pytest

import plotting
from run_beamplasma import weakbeam_instability
from run_coldplasma import cold_plasma_oscillations
from run_twostream import two_stream_instability
from run_wave import wave_propagation

np.random.seed(0)


@pytest.mark.parametrize(["filename", ], [["BP1", ]])
def test_beamplasma(filename):
    s = weakbeam_instability(filename, save_data=True)
    plotting.plotting(s, show=False, save=True, animate=True)


@pytest.mark.parametrize(["filename", "plasma_frequency", "push_mode", "N_electrons", "NG", "qmratio"],
                         [
                             ("CO1", 1, 2, 1024, 64, -1),
                             ])
def test_coldplasma(filename, plasma_frequency, push_mode, N_electrons, NG, qmratio):
    s = cold_plasma_oscillations(filename, qmratio=qmratio, plasma_frequency=plasma_frequency, NG=NG,
                                 N_electrons=N_electrons, push_mode=push_mode, save_data=True)
    plotting.plotting(s, show=False, save=True, animate=True)


@pytest.mark.parametrize(["filename", "plasma_frequency", "NT", "dt", "N_electrons", "NG", "qmratio", "vrandom"],
                         [
                             ("TS1", 1, 1500, 0.04, 1024, 64, -1, 0),
                             ("TS2", 5, 1500, 0.04, 1024, 64, -1, 0),
                             ("TS3", 10, 1500, 0.04, 1024, 64, -1, 0),
                             ("TSRANDOM1", 1, 1500, 0.04, 1024, 64, -1, 0.1),
                             ("TSRANDOM2", 5, 1500, 0.04, 1024, 64, -1, 0.1),
                             ("TSRANDOM3", 10, 1500, 0.04, 1024, 64, -1, 0.1),
                             ])
def test_twostream(filename, plasma_frequency, NT, dt, N_electrons, NG, qmratio, vrandom):
    s = two_stream_instability(filename, qmratio=qmratio, plasma_frequency=plasma_frequency, NG=NG,
                               N_electrons=N_electrons, NT=NT, dt=dt, vrandom=vrandom, save_data=True)
    plotting.plotting(s, show=False, save=True, animate=True)


@pytest.mark.parametrize(["filename", "bc", "bc_parameter_function", "bc_params", "polarization_angle"],
                         [
                             ("sin1", "sine", lambda t: t / 25, (1,), 0),
                             ("sin1_polarized", "sine", lambda t: t / 25, (1,), np.pi/3),
                             ("laser2", "laser", lambda t: t / 25, (1, 2), 0),
                             ("laser2_polarized", "laser", lambda t: t / 25, (1, 2), 2*np.pi/3),
                             ("laser6", "laser", lambda t: t / 25, (1, 6), 0),
                             ("laser6_polarized", "laser", lambda t: t / 25, (1, 6), 2*np.pi/3),
                             ])
def test_wave_propagation(filename, bc, bc_parameter_function, bc_params, polarization_angle):
    s = wave_propagation(filename, bc, bc_parameter_function)
    plotting.plotting(s, show=False, save=True, animate=True)
