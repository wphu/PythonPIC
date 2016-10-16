import h5py

filename = "test.hdf5"
f = h5py.File(filename, "r")
for i in f.items():
    print(i)
for i in f.attrs:
    print(i, f.attrs[i])
