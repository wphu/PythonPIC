import argparse
import static_plots
import animation
import Simulation
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("filename", help="hdf5 file name for storing data")
parser.add_argument('-dev', action='store_true')
args = parser.parse_args()

if(args.filename[-5:] != ".hdf5"):
    args.filename = args.filename + ".hdf5"

S = Simulation.load_data(args.filename)

directory = "data_analysis/"

static_plots.energy_time_plots(S, directory + args.filename.replace(".hdf5", ".png"))
if args.dev:
    videofile_name = None
else:
    videofile_name = directory + args.filename.replace(".hdf5", ".mp4")
animation.animation(S, videofile_name)
plt.show()
