import h5py
import pandas as pd
file = "data_analysis/laser-shield/v21/production_run_23_0/production_run_23_0.hdf5"
df = pd.DataFrame()
with h5py.File(file) as f:
    kinetic = f['species']['electrons']['Kinetic energy']
    field = f['grid']['Efield']
    kin = pd.DataFrame(kinetic[...])
    ex = pd.DataFrame(field[:,:,0])
    ey = pd.DataFrame(field[:,:,1])
    ez = pd.DataFrame(field[:,:,2])
    attrs = f['grid'].attrs
    epsilon_0 = attrs['epsilon_0']
    for key, value in attrs.items():
        print(key, value)
