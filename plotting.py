import argparse
import static_plots
import animation
import Simulation
import matplotlib.pyplot as plt

directory = "data_analysis/"

def plotting(filename, show = True, save = False, lines=False):
    print("Plotting for %s" %filename)
    S = Simulation.load_data(filename)
    static_plots.energy_time_plots(S, filename.replace(".hdf5", "_energy.png"))
    static_plots.ESE_time_plots(S, filename.replace(".hdf5", "_mode_energy.png"))
    static_plots.temperature_time_plot(S, filename.replace(".hdf5", "_temperature.png"))
    if save:
        videofile_name = filename.replace(".hdf5", ".mp4")
    else:
        videofile_name = None
    animation.animation(S, videofile_name, lines)
    if show:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="hdf5 file name for storing data")
    parser.add_argument('-save', action='store_false')
    parser.add_argument('-lines', action='store_true')
    args = parser.parse_args()
    if(args.filename[-5:] != ".hdf5"):
        args.filename = args.filename + ".hdf5"

    plotting(args.filename, show = True, save = args.save, lines = args.lines)
