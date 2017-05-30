"""Animates the simulation to show quantities that change over time"""
# coding=utf-8

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

from .time_snapshots import FrequencyPlot, PhasePlot, SpatialDistributionPlot, IterationCounter, \
    TripleFieldPlot, TripleCurrentPlot
from ..algorithms import helper_functions


# formatter = matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False)


def animation(S, save: bool = False, alpha=1, frame_to_draw="animation", writer="ffmpeg"):
    """

    Creates an animation from `Plot`s.

    Parameters
    ----------
    S : Simulation
        Data source
    save : bool
        Whether to save the simulation
    alpha : float
        Opacity value from 0 to 1, used in the phase plot.
    frame_to_draw : str, list or int
        Default value is "animation" - this causes the full animation to be played.
        A list such as [0, 10, 100, 500, 1000] causes the animation to plot iterations 0, 10, 100... etc
        and save them as pictures. Does not display the frames.
        An integer causes the animation to display a single iterations, optionally saving it as a picture.
    writer : str
        A Matplotlib writer string.
    Returns
    -------
    figure or matplotlib animation
        Plot object, depending on `frame_to_draw`.
    """
    assert alpha <= 1, "alpha too large!"
    assert alpha >= 0, "alpha must be between 0 and 1!"
    """ animates the simulation, showing:
    * grid charge vs grid position
    * grid electric field vs grid position
    * particle phase plot (velocity vs position)
    * spatial energy modes

    S - Simulation object with run's data
    videofile_name - should be in format FILENAME.mp4;
        if not None, saves to file
    lines - boolean flag; draws particle trajectories on phase plot
    # TODO: investigate lines flag in animation
    alpha - float (0, 1) - controls opacity for phase plot

    returns: matplotlib figure with animation
    """
    S.postprocess()
    fig = plt.figure(figsize=(13, 10))
    charge_axis = fig.add_subplot(421)
    current_axes = [fig.add_subplot(423 + 2 * i) for i in range(3)]
    phase_axes_x = fig.add_subplot(422)
    phase_axes_y = fig.add_subplot(424)
    phase_axes_z = fig.add_subplot(426)
    freq_axes = fig.add_subplot(428)

    fig.suptitle(str(S), fontsize=12)
    fig.subplots_adjust(top=0.81, bottom=0.08, left=0.15, right=0.95,
                        wspace=.25, hspace=0.6)  # REFACTOR: remove particle windows if there are no particles

    phase_plot_x = PhasePlot(S, phase_axes_x, "x", "v_x", alpha)
    phase_plot_y = PhasePlot(S, phase_axes_y, "x", "v_y", alpha)
    phase_plot_z = PhasePlot(S, phase_axes_z, "x", "v_z", alpha)
    # velocity_histogram = Histogram(S, distribution_axes, "v_x")
    freq_plot = FrequencyPlot(S, freq_axes)
    charge_plot = SpatialDistributionPlot(S, charge_axis)
    iteration = IterationCounter(S, freq_axes)
    current_plots = TripleCurrentPlot(S, current_axes)
    field_plots = TripleFieldPlot(S, [current_ax.twinx() for current_ax in current_axes])

    plots = [
            phase_plot_x,
            phase_plot_y,
            phase_plot_z,
             freq_plot,
             charge_plot,
             iteration,
             current_plots,
             field_plots,
        ]

    results = []
    for plot in plots:
        for result in plot.return_animated():
            results.append(result)

    def animate(i, verbose=False):
        """draws the i-th frame of the simulation"""
        if verbose:
            helper_functions.report_progress(i, S.grid.NT)
        for plot in plots:
            plot.update(i)
        return results
    if isinstance(frame_to_draw, np.ndarray):
        frame_to_draw = list(frame_to_draw)
    frames = np.arange(0, S.NT,
                       helper_functions.calculate_particle_iter_step(S.NT),
                       dtype=int)
    if frame_to_draw == "animation":
        print("Drawing full animation.")
        mpl_Writer = anim.writers[writer]
        mpl_writer = mpl_Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        def init():
            """initializes animation window for faster drawing"""
            for plot in plots:
                plot.animation_init()
            return results


        # noinspection PyTypeChecker
        animation_object = anim.FuncAnimation(fig, animate, interval=100,
                                              frames=frames,
                                              blit=True, init_func=init,
                                              fargs=(save,))
        if save:
            helper_functions.make_sure_path_exists(S.filename)
            videofile_name = S.filename.replace(".hdf5", ".mp4")
            print(f"Saving animation to {videofile_name}")
            animation_object.save(videofile_name, writer=mpl_writer)#, extra_args=['-vcodec', 'libx264'])
            print(f"Saved animation to {videofile_name}")
        return animation_object
    elif frame_to_draw == "anim_snapshots":
        print("Drawing animation as snapshots.")
        for i in frames:
            animate(i)
            helper_functions.make_sure_path_exists(S.filename)
            file_name = S.filename.replace(".hdf5", f"_{i:06}.png")
            print(f"Saving iteration {i} to {file_name}")
            fig.savefig(file_name)
        return fig
    elif isinstance(frame_to_draw, list):
        print("Drawing frames." + str(frame_to_draw))
        for i in frame_to_draw:
            animate(i)
            helper_functions.make_sure_path_exists(S.filename)
            file_name = S.filename.replace(".hdf5", f"_{i:06}.png")
            print(f"Saving iteration {i} to {file_name}")
            fig.savefig(file_name)
        return fig
    elif isinstance(frame_to_draw, int):
        print("Drawing iteration", frame_to_draw)
        animate(frame_to_draw)
        if save:
            helper_functions.make_sure_path_exists(S.filename)
            file_name = S.filename.replace(".hdf5", f"_{frame_to_draw}.png")
            print(f"Saving iteration {frame_to_draw} to {file_name}")
            fig.savefig(file_name)
        return fig
    else:
        raise ValueError("Incorrect frame_to_draw - must be 'animation' or number of iteration to draw.")
