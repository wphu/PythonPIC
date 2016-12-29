import numpy as np
import matplotlib.pyplot as plt
import Simulation
from spectro2 import dispersion_relation

for i in range(1,11):
    filename = "data_analysis/CO{}.hdf5".format(i)
    S = Simulation.load_data(filename)
    t = np.arange(S.NT+1)*S.dt
    dispersion_relation(t, S.grid.x, S.grid.electric_field_history, plot_spectro=True)
    #TODO: multiple maxima due to noise?
