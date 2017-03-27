# coding=utf-8
import numpy as np
import pytest

from run_twostream import two_stream_instability


@pytest.mark.parametrize(["NG", "N_electrons"], [
    (64, 512),
    (128, 1024),
    ])
def test_linear_regime_beam_stability(NG, N_electrons):
    run_name = f"TS_LINEAR_{NG}_{N_electrons}"
    S = two_stream_instability(run_name,
                               NG=NG,
                               N_electrons=N_electrons,
                               save_data=False,
                               )
    assert (~did_it_thermalize(S)).all()


@pytest.mark.parametrize(["NG", "N_electrons", "plasma_frequency"], [
    (64, 1024, 5),
    (128, 2048, 5),
    (64, 1024, 7),
    (64, 1024, 4),
    ])
def test_nonlinear_regime_beam_instability(NG, N_electrons, plasma_frequency):
    run_name = f"TS_NONLINEAR_{NG}_{N_electrons}_{plasma_frequency}"
    S = two_stream_instability(run_name,
                               NG=NG,
                               N_electrons=N_electrons,
                               plasma_frequency=plasma_frequency,
                               dt=0.2 / 2,
                               NT=300 * 2,
                               save_data=False,
                               )
    assert did_it_thermalize(S).all()


def did_it_thermalize(S):
    initial_velocities = np.array([s.velocity_history[0, :, 0].mean() for s in S.list_species])
    initial_velocity_stds = np.array([s.velocity_history[0, :, 0].std() for s in S.list_species])
    average_velocities = np.array([s.velocity_history[:, :, 0].mean() for s in S.list_species])
    return np.abs((initial_velocities - average_velocities)) > initial_velocity_stds
