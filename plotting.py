import argparse
import static_plots
import animation
import Simulation
import matplotlib.pyplot as plt

directory = "data_analysis/"

def plotting(filename, save_directory = directory, show = True, dev = True, lines=False):
    S = Simulation.load_data(filename)
    static_plots.energy_time_plots(S, save_directory + filename.replace(".hdf5", ".png"))
    if dev:
        videofile_name = None
    else:
        videofile_name = save_directory + filename.replace(".hdf5", ".mp4")
    animation.animation(S, videofile_name, lines)
    if show:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="hdf5 file name for storing data")
    parser.add_argument('-dev', action='store_true')
    parser.add_argument('-lines', action='store_true')
    args = parser.parse_args()
    if(args.filename[-5:] != ".hdf5"):
        args.filename = args.filename + ".hdf5"

    plotting(args.filename, directory, show = True, dev = args.dev, lines = args.lines)
