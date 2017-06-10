import h5py
import pandas as pd
file = "data_analysis/laser-shield/v19/production_run_22_0/production_run_22_0.hdf5"
df = pd.DataFrame()
with h5py.File(file) as f:
    field = f['grid']['Efield']
    ex = pd.DataFrame(field[:,:,0])
    ey = pd.DataFrame(field[:,:,1])
    ez = pd.DataFrame(field[:,:,2])

print(ex)