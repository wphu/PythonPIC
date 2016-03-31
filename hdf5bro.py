import h5py

filename = "test.hdf5"
with h5py.File(filename, "r") as f:
    for i in f.items():
        print(i)
    for i in f.attrs:
        print(i, f.attrs[i])
