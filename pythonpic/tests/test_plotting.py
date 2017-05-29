# coding=utf-8
import pytest
import os
from matplotlib import animation as mpl_anim
import numpy as np
from time import time

from ..algorithms import helper_functions
from ..configs import cold_plasma_oscillations
from ..visualization.plotting import plots
from ..visualization.animation import animation


@pytest.fixture(scope="module")
def helper_short_simulation():
    if "DISPLAY" not in os.environ.keys():
        print("Not running display test right now.")
        return False
    else:
        run_name = "visualization_test"
        S = cold_plasma_oscillations(run_name, save_data=False)
        return S


def test_static_plots(helper_short_simulation):
    S = helper_short_simulation
    if S:
        try:
            plots(S, save_static=True)
        except:
            assert False, "Failure on saving static plot"

def test_animation(helper_short_simulation):
    S = helper_short_simulation
    if S:
        try:
            plots(S, save_animation=True)
        except:
            assert False, "Failure on saving animation"


def test_writer_manual_speed(helper_short_simulation):
    S = helper_short_simulation
    start_time = time()
    frames = list(np.arange(0, S.NT,
                       helper_functions.calculate_particle_iter_step(S.NT),
                       dtype=int))
    animation(S, save=True, frame_to_draw=frames)
    endtime = time()
    runtime = endtime - start_time
    print(runtime)
    assert runtime

# @pytest.mark.parametrize("writer", ['ffmpeg', 'ffmpeg_file', 'mencoder'])
# def test_writer_speed(helper_short_simulation, writer):
#     S = helper_short_simulation
#     start_time = time()
#     animation(S, save=True, writer=writer)
#     endtime = time()
#     runtime = endtime - start_time
#     print(writer, runtime)
#     assert runtime
