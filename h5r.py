import h5py
import pandas as pd
import numpy as np
file = "data_analysis/laser-shield/v21/production_run_23_0/production_run_23_0.hdf5"
file = "data_analysis/laser-shield/v21/production_run_22_0/production_run_22_0.hdf5"
# file = "data_analysis/laser-shield/v21/10000_run_22_0/10000_run_22_0.hdf5"
df = pd.DataFrame()
with h5py.File(file) as f:
    kinetic = f['species']['electrons']['Kinetic energy']
    efield = f['grid']['Efield']
    bfield = f['grid']['Efield']
    kin = pd.DataFrame(kinetic[...])
    ex = pd.DataFrame(efield[:,:,0])
    ey = pd.DataFrame(efield[:,:,1])
    ez = pd.DataFrame(efield[:,:,2])
    bx = pd.DataFrame(bfield[:,:,0])
    by = pd.DataFrame(bfield[:,:,1])
    bz = pd.DataFrame(bfield[:,:,2])
    attrs = f['grid'].attrs
    epsilon_0 = attrs['epsilon_0']
    c = attrs['c']
    NG = attrs['NGrid']
    L = attrs['L']
    for key, value in attrs.items():
        print(key, value)
    print("electrons")
    electron_attrs = f['species']['electrons'].attrs
    m = electron_attrs['m']
    q = electron_attrs['q']
    for key, value in electron_attrs.items():
        print(key, value)

Bmax = np.sqrt(bx**2 + by**2 + bz**2).max().max()
stability_parameter = abs(m * NG * c / q / Bmax / L)
print("STABILITY", stability_parameter)