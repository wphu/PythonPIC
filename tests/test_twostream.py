# coding=utf-8
import pytest

from helper_functions import did_it_thermalize
from plotting import plotting
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
    assert (~did_it_thermalize(S)).all(), plotting(S, show=True, save=True, animate=True)


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
    assert did_it_thermalize(S).all(), plotting(S, show=True, save=True, animate=True)


@pytest.mark.parametrize(["v0", "NT"], [
    (1, 450),
    (2, 450),
    (3, 300),
    # TEST: this needs an xfail
    ])
def test_electron_positron(v0, NT):
    """the electron-positron run is much noisier
    the particles do not really seem to jump between beams """
    S = two_stream_instability("TS_EP",
                               NG=64,
                               N_electrons=512,
                               plasma_frequency=5,
                               dt=0.2,
                               NT=NT,
                               v0=v0,
                               species_2_sign=-1)
    average_velocities = [sp.velocity_history[:, int(sp.N / 2), 0].mean() for sp in S.list_species]
    avg_velocity_difference = abs(average_velocities[1] - average_velocities[0])
    print(avg_velocity_difference)
    assert avg_velocity_difference > v0, plotting(S, show=True, save=True, animate=True)


# @pytest.mark.parametrize(["push_amplitude"], [
#    (0.95,), (1.05,), (1.25,),
#    ])
# def test_push_amplitude(push_amplitude):
#     S = two_stream_instability(f"TS_PUSH_{push_amplitude}",
#                            NG=64,
#                            N_electrons=512,
#                            push_amplitude=push_amplitude,
#                            )
#
#     # TEST: finish this

