import argparse
import static_plots
import animation
import Simulation
import matplotlib.pyplot as plt
from spectrograph import spectral_analysis

directory = "data_analysis/"

def plotting(filename, show = True, save = False, lines=False, alpha=1):
    print("Plotting for %s" %filename)
    S = Simulation.load_data(filename)
    static_plots.energy_time_plots(S, filename.replace(".hdf5", "_energy.png"))
    static_plots.ESE_time_plots(S, filename.replace(".hdf5", "_mode_energy.png"))
    static_plots.temperature_time_plot(S, filename.replace(".hdf5", "_temperature.png"))
    spectral_analysis(S, filename.replace(".hdf5","_spectro.png"))
    if save:
        videofile_name = filename.replace(".hdf5", ".mp4")
    else:
        videofile_name = None
    anim = animation.animation(S, videofile_name, lines, alpha=alpha)
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close("all")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="hdf5 file name for storing data")
    parser.add_argument('-save', action='store_false')
    parser.add_argument('-lines', action='store_true')
    args = parser.parse_args()
    if(args.filename[-5:] != ".hdf5"):
        args.filename = args.filename + ".hdf5"

    plotting(args.filename, show = True, save = args.save, lines = args.lines)
